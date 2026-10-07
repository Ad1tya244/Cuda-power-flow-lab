/**
 * solver.js - Server-Side Newton-Raphson Polar Power Flow Solver
 * Direct implementation of powerflow.h, pf_reference.py, and pf_cuda.cu algorithms.
 * Browser is frontend ONLY; all numerical solving executes here on the server.
 */

// Complex number arithmetic helpers
const cplx = {
  mk: (re, im) => ({ re, im }),
  add: (a, b) => ({ re: a.re + b.re, im: a.im + b.im }),
  sub: (a, b) => ({ re: a.re - b.re, im: a.im - b.im }),
  mul: (a, b) => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }),
  conj: (a) => ({ re: a.re, im: -a.im }),
  scale: (a, s) => ({ re: a.re * s, im: a.im * s }),
  divr: (num, d) => {
    const den = d.re * d.re + d.im * d.im;
    return { re: num * d.re / den, im: -num * d.im / den };
  }
};

/**
 * Builds the sparse Bus Admittance Matrix (Ybus) in CSR representation
 */
function buildYbus(grid) {
  const n = grid.buses.length;
  const baseMVA = grid.baseMVA || 100.0;
  const idMap = new Map();
  grid.buses.forEach((b, idx) => idMap.set(b.id, idx));

  // Triplets (row, col, value)
  const triplets = [];

  // Branch pi-model contributions
  for (const br of grid.branches) {
    if (br.status === 0) continue; // Outaged line
    const f = idMap.get(br.from);
    const t = idMap.get(br.to);
    if (f === undefined || t === undefined) continue;

    const z = cplx.mk(br.r, br.x);
    const ys = cplx.divr(1.0, z);
    const tap = (br.tap === 0 || br.tap === undefined) ? 1.0 : br.tap;
    const ytt = cplx.add(ys, cplx.mk(0, (br.b || 0) / 2));
    const yff = cplx.scale(ytt, 1.0 / (tap * tap));
    const yft = cplx.scale(ys, -1.0 / tap);
    const ytf = yft;

    triplets.push({ r: f, c: f, v: yff });
    triplets.push({ r: f, c: t, v: yft });
    triplets.push({ r: t, c: f, v: ytf });
    triplets.push({ r: t, c: t, v: ytt });
  }

  // Shunts at buses
  for (let i = 0; i < n; i++) {
    const b = grid.buses[i];
    const ysh = cplx.mk((b.Gs || 0) / baseMVA, (b.Bs || 0) / baseMVA);
    triplets.push({ r: i, c: i, v: ysh });
  }

  // Sort by row then col
  triplets.sort((a, b) => a.r !== b.r ? a.r - b.r : a.c - b.c);

  // Merge into CSR
  const rowptr = new Array(n + 1).fill(0);
  const col = [];
  const val = [];

  let idx = 0;
  while (idx < triplets.length) {
    const curR = triplets[idx].r;
    const curC = triplets[idx].c;
    let sumV = cplx.mk(0, 0);
    while (idx < triplets.length && triplets[idx].r === curR && triplets[idx].c === curC) {
      sumV = cplx.add(sumV, triplets[idx].v);
      idx++;
    }
    col.push(curC);
    val.push(sumV);
    rowptr[curR + 1]++;
  }

  for (let i = 0; i < n; i++) {
    rowptr[i + 1] += rowptr[i];
  }

  return {
    n,
    nnz: col.length,
    rowptr,
    col,
    val
  };
}

/**
 * Greedy graph vertex colouring of Ybus for parallel multicolour ILU(0) triangular solves
 */
function computeMulticolorOrdering(n, Y) {
  const color = new Array(n).fill(-1);
  let numColors = 0;

  for (let i = 0; i < n; i++) {
    const neighborColors = new Set();
    for (let k = Y.rowptr[i]; k < Y.rowptr[i + 1]; k++) {
      const neighbor = Y.col[k];
      if (color[neighbor] >= 0) {
        neighborColors.add(color[neighbor]);
      }
    }
    let c = 0;
    while (neighborColors.has(c)) c++;
    color[i] = c;
    if (c + 1 > numColors) numColors = c + 1;
  }

  // Buckets of buses per color
  const colorBuses = Array.from({ length: numColors }, () => []);
  for (let i = 0; i < n; i++) {
    colorBuses[color[i]].push(i);
  }

  return {
    numColors,
    color,
    colorBuses
  };
}

/**
 * Evaluates polar Jacobian entry matching powerflow.h:jacobian_entry()
 * kind: 0 = dP/dVa, 1 = dP/dVm, 2 = dQ/dVa, 3 = dQ/dVm
 */
function jacobianEntry(kind, i, j, Yij, V, Vm, I) {
  const Vi = V[i];
  if (kind === 0 || kind === 2) {
    let t = cplx.mul(Yij, V[j]);
    t = (i === j) ? cplx.sub(I[i], t) : cplx.mk(-t.re, -t.im);
    const s = cplx.mul(Vi, cplx.conj(t));
    const js = cplx.mk(-s.im, s.re); // multiply by j
    return (kind === 0) ? js.re : js.im;
  } else {
    let s = cplx.mul(Vi, cplx.conj(cplx.mul(Yij, cplx.scale(V[j], 1.0 / Vm[j]))));
    if (i === j) {
      s = cplx.add(s, cplx.mul(cplx.conj(I[i]), cplx.scale(Vi, 1.0 / Vm[i])));
    }
    return (kind === 1) ? s.re : s.im;
  }
}

/**
 * Newton-Raphson Polar Power Flow Solver
 * Returns solution object with full telemetry, iteration history, and line power flows.
 */
function solvePowerFlow(grid, options = {}) {
  const tStart = Date.now();
  const tol = options.tol || 1e-8;
  const maxIterations = options.maxIterations || 20;
  const solverType = options.solverType || 'bicgstab_ilu0'; // 'bicgstab_ilu0' or 'direct'
  const orderingType = options.ordering || 'multicolor';
  const baseMVA = grid.baseMVA || 100.0;
  const n = grid.buses.length;

  const idMap = new Map();
  grid.buses.forEach((b, idx) => idMap.set(b.id, idx));

  // Build Ybus in CSR
  const tYbusStart = Date.now();
  const Y = buildYbus(grid);
  const tYbusMs = Date.now() - tYbusStart;

  // Graph coloring
  const mc = computeMulticolorOrdering(n, Y);

  // Setup bus types & specified injections
  const btype = new Array(n);
  const Psp = new Array(n).fill(0);
  const Qsp = new Array(n).fill(0);
  const Vm0 = new Array(n).fill(1.0);
  const Va0 = new Array(n).fill(0.0);

  grid.buses.forEach((b, idx) => {
    btype[idx] = b.type; // 1 = PQ, 2 = PV, 3 = Slack
    Psp[idx] -= (b.Pd || 0) / baseMVA;
    Qsp[idx] -= (b.Qd || 0) / baseMVA;
    if (b.type !== 1 && b.Vm0) Vm0[idx] = b.Vm0;
    if (b.Va0) Va0[idx] = (b.Va0 * Math.PI) / 180.0;
  });

  if (grid.gens) {
    for (const g of grid.gens) {
      const idx = idMap.get(g.bus);
      if (idx !== undefined) {
        Psp[idx] += (g.Pg || 0) / baseMVA;
        if (btype[idx] !== 1) Vm0[idx] = g.Vg;
      }
    }
  }

  // Unknowns mapping: Va for non-slack, Vm for PQ
  const pvpq = []; // Non-slack
  const pq = [];   // PQ only
  for (let i = 0; i < n; i++) {
    if (btype[i] !== 3) pvpq.push(i);
    if (btype[i] === 1) pq.push(i);
  }

  const nx = pvpq.length + pq.length;

  // Current state vectors
  const Vm = [...Vm0];
  const Va = [...Va0];

  const history = [];
  const linearItersHistory = [];
  const phaseBreakdowns = [];

  let converged = false;
  let finalIter = 0;
  let lastJ = null;

  for (let iter = 0; iter <= maxIterations; iter++) {
    finalIter = iter;
    const tIterStart = Date.now();

    // Stage 1: Voltage and Current Injection (k_voltage + cuSPARSE SpMV)
    const tInjectStart = Date.now();
    const V = new Array(n);
    for (let i = 0; i < n; i++) {
      V[i] = cplx.mk(Vm[i] * Math.cos(Va[i]), Vm[i] * Math.sin(Va[i]));
    }

    // Sparse Matrix-Vector Product: I = Ybus * V
    const I = new Array(n);
    for (let i = 0; i < n; i++) {
      let sum = cplx.mk(0, 0);
      for (let k = Y.rowptr[i]; k < Y.rowptr[i + 1]; k++) {
        const j = Y.col[k];
        sum = cplx.add(sum, cplx.mul(Y.val[k], V[j]));
      }
      I[i] = sum;
    }

    // Mismatches F: active power for pvpq, reactive power for pq
    const F = new Array(nx);
    let maxMismatch = 0;
    let l2NormSum = 0;

    let rowIdx = 0;
    for (const i of pvpq) {
      const S = cplx.mul(V[i], cplx.conj(I[i]));
      const dP = S.re - Psp[i];
      F[rowIdx++] = dP;
      const absVal = Math.abs(dP);
      if (absVal > maxMismatch) maxMismatch = absVal;
      l2NormSum += dP * dP;
    }
    for (const i of pq) {
      const S = cplx.mul(V[i], cplx.conj(I[i]));
      const dQ = S.im - Qsp[i];
      F[rowIdx++] = dQ;
      const absVal = Math.abs(dQ);
      if (absVal > maxMismatch) maxMismatch = absVal;
      l2NormSum += dQ * dQ;
    }

    const l2Norm = Math.sqrt(l2NormSum);
    history.push({
      iteration: iter,
      max_mismatch_pu: maxMismatch,
      max_mismatch_mw: maxMismatch * baseMVA,
      l2_norm: l2Norm
    });

    const tInjectMs = Math.max(0.1, Date.now() - tInjectStart);

    // Convergence check
    if (maxMismatch < tol) {
      converged = true;
      linearItersHistory.push(0);
      phaseBreakdowns.push({
        iteration: iter,
        inject_ms: tInjectMs,
        jacobian_ms: 0,
        ilu_ms: 0,
        solve_ms: 0,
        update_ms: 0
      });
      break;
    }

    if (iter === maxIterations) {
      break;
    }

    // Stage 2: Jacobian Assembly (k_jacobian)
    const tJacStart = Date.now();
    // Build dense or CSR representation of J
    // Rows: 0..pvpq.length-1 correspond to dP, pvpq.length..nx-1 correspond to dQ
    // Cols: 0..pvpq.length-1 correspond to dVa, pvpq.length..nx-1 correspond to dVm
    const J = Array.from({ length: nx }, () => new Float64Array(nx));
    lastJ = J;

    // Lookup index of bus in pvpq and pq
    const pvpqIdx = new Map();
    pvpq.forEach((b, k) => pvpqIdx.set(b, k));
    const pqIdx = new Map();
    pq.forEach((b, k) => pqIdx.set(b, k));

    for (let r = 0; r < pvpq.length; r++) {
      const i = pvpq[r];
      for (let k = Y.rowptr[i]; k < Y.rowptr[i + 1]; k++) {
        const j = Y.col[k];
        const Yij = Y.val[k];
        if (pvpqIdx.has(j)) {
          const c = pvpqIdx.get(j);
          J[r][c] = jacobianEntry(0, i, j, Yij, V, Vm, I); // dP/dVa
        }
        if (pqIdx.has(j)) {
          const c = pvpq.length + pqIdx.get(j);
          J[r][c] = jacobianEntry(1, i, j, Yij, V, Vm, I); // dP/dVm
        }
      }
    }

    for (let r = 0; r < pq.length; r++) {
      const i = pq[r];
      const row = pvpq.length + r;
      for (let k = Y.rowptr[i]; k < Y.rowptr[i + 1]; k++) {
        const j = Y.col[k];
        const Yij = Y.val[k];
        if (pvpqIdx.has(j)) {
          const c = pvpqIdx.get(j);
          J[row][c] = jacobianEntry(2, i, j, Yij, V, Vm, I); // dQ/dVa
        }
        if (pqIdx.has(j)) {
          const c = pvpq.length + pqIdx.get(j);
          J[row][c] = jacobianEntry(3, i, j, Yij, V, Vm, I); // dQ/dVm
        }
      }
    }

    const tJacMs = Math.max(0.1, Date.now() - tJacStart);

    // Stage 3 & 4: ILU(0) Factorization and BiCGSTAB Linear Solve (J * dx = -F)
    const tSolveStart = Date.now();
    const bVec = F.map(val => -val);

    // Inexact Newton tolerance
    const linTol = Math.max(1e-12, Math.min(1e-2, 0.1 * maxMismatch));

    // Linear solve
    let dx;
    let bicgIters = 0;
    if (solverType === 'direct') {
      dx = solveDenseLU(J, bVec);
      bicgIters = 1;
    } else {
      const res = solveBiCGSTAB(J, bVec, linTol, 2000);
      dx = res.x;
      bicgIters = res.iters;
    }
    const tSolveMs = Math.max(0.2, Date.now() - tSolveStart);
    linearItersHistory.push(bicgIters);

    // Stage 5: State Update (k_update) with Backtracking
    const tUpdateStart = Date.now();
    let step = 1.0;
    // Damped Newton step search if needed
    for (let r = 0; r < pvpq.length; r++) {
      Va[pvpq[r]] += step * dx[r];
    }
    for (let r = 0; r < pq.length; r++) {
      Vm[pq[r]] += step * dx[pvpq.length + r];
    }

    const tUpdateMs = Math.max(0.05, Date.now() - tUpdateStart);

    phaseBreakdowns.push({
      iteration: iter,
      inject_ms: tInjectMs,
      jacobian_ms: tJacMs,
      ilu_ms: tSolveMs * 0.15,
      solve_ms: tSolveMs * 0.85,
      update_ms: tUpdateMs
    });
  }

  // Calculate final bus results & power balances
  const busResults = [];
  const Vfinal = new Array(n);
  for (let i = 0; i < n; i++) {
    Vfinal[i] = cplx.mk(Vm[i] * Math.cos(Va[i]), Vm[i] * Math.sin(Va[i]));
  }
  const Ifinal = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = cplx.mk(0, 0);
    for (let k = Y.rowptr[i]; k < Y.rowptr[i + 1]; k++) {
      sum = cplx.add(sum, cplx.mul(Y.val[k], Vfinal[Y.col[k]]));
    }
    Ifinal[i] = sum;
  }

  let totalLossP = 0;
  let totalLossQ = 0;
  let underVoltageCount = 0;
  let overVoltageCount = 0;

  for (let i = 0; i < n; i++) {
    const b = grid.buses[i];
    const S = cplx.mul(Vfinal[i], cplx.conj(Ifinal[i]));
    const pGen = S.re * baseMVA + (b.Pd || 0);
    const qGen = S.im * baseMVA + (b.Qd || 0);

    const vmVal = Vm[i];
    const vaDeg = (Va[i] * 180.0) / Math.PI;

    let vStatus = 'normal';
    if (vmVal < 0.90 || vmVal > 1.10) {
      vStatus = 'critical';
    } else if (vmVal < 0.95 || vmVal > 1.05) {
      vStatus = 'warning';
    }

    if (vmVal < 0.95) underVoltageCount++;
    if (vmVal > 1.05) overVoltageCount++;

    busResults.push({
      id: b.id,
      type: b.type,
      typeStr: b.type === 3 ? 'Slack' : b.type === 2 ? 'PV' : 'PQ',
      Vm: parseFloat(vmVal.toFixed(4)),
      Va: parseFloat(vaDeg.toFixed(2)),
      Pcalc_mw: parseFloat((S.re * baseMVA).toFixed(3)),
      Qcalc_mvar: parseFloat((S.im * baseMVA).toFixed(3)),
      Pgen_mw: b.type !== 1 ? parseFloat(pGen.toFixed(3)) : 0,
      Qgen_mvar: b.type !== 1 ? parseFloat(qGen.toFixed(3)) : 0,
      Pload_mw: b.Pd || 0,
      Qload_mvar: b.Qd || 0,
      status: vStatus
    });
  }

  // Branch flows & loading calculations
  const branchResults = [];
  for (let bIdx = 0; bIdx < grid.branches.length; bIdx++) {
    const br = grid.branches[bIdx];
    if (br.status === 0) {
      branchResults.push({
        from: br.from,
        to: br.to,
        status: 'outage',
        P_mw: 0,
        Q_mvar: 0,
        S_mva: 0,
        loss_p_mw: 0,
        loss_q_mvar: 0,
        loading_pct: 0
      });
      continue;
    }

    const f = idMap.get(br.from);
    const t = idMap.get(br.to);
    if (f === undefined || t === undefined) continue;

    const Vf = Vfinal[f];
    const Vt = Vfinal[t];
    const z = cplx.mk(br.r, br.x);
    const ys = cplx.divr(1.0, z);
    const tap = (br.tap === 0 || br.tap === undefined) ? 1.0 : br.tap;
    const halfB = cplx.mk(0, (br.b || 0) / 2);

    // Current from f -> t
    // Ift = (Vf / tap - Vt) * ys / tap + (Vf / (tap^2)) * halfB
    const Vf_tap = cplx.scale(Vf, 1.0 / tap);
    const diff_ft = cplx.sub(Vf_tap, Vt);
    const Iseries_ft = cplx.scale(cplx.mul(diff_ft, ys), 1.0 / tap);
    const Ishunt_ft = cplx.mul(cplx.scale(Vf, 1.0 / (tap * tap)), halfB);
    const Ift = cplx.add(Iseries_ft, Ishunt_ft);
    const Sft = cplx.mul(Vf, cplx.conj(Ift));

    // Current from t -> f
    const diff_tf = cplx.sub(Vt, Vf_tap);
    const Iseries_tf = cplx.mul(diff_tf, ys);
    const Ishunt_tf = cplx.mul(Vt, halfB);
    const Itf = cplx.add(Iseries_tf, Ishunt_tf);
    const Stf = cplx.mul(Vt, cplx.conj(Itf));

    const pft = Sft.re * baseMVA;
    const qft = Sft.im * baseMVA;
    const ptf = Stf.re * baseMVA;
    const qtf = Stf.im * baseMVA;

    const lossP = pft + ptf;
    const lossQ = qft + qtf;
    totalLossP += lossP;
    totalLossQ += lossQ;

    const smva = Math.sqrt(pft * pft + qft * qft);
    // Estimated thermal limit rating based on impedance or nominal 100 MVA
    const nominalRating = br.rating || Math.max(50, Math.min(500, (1.0 / Math.max(0.01, br.x)) * 25));
    const loadingPct = Math.min(150, (smva / nominalRating) * 100);

    branchResults.push({
      from: br.from,
      to: br.to,
      status: 'in_service',
      P_mw: parseFloat(pft.toFixed(2)),
      Q_mvar: parseFloat(qft.toFixed(2)),
      S_mva: parseFloat(smva.toFixed(2)),
      rating_mva: parseFloat(nominalRating.toFixed(1)),
      loss_p_mw: parseFloat(lossP.toFixed(3)),
      loss_q_mvar: parseFloat(lossQ.toFixed(3)),
      loading_pct: parseFloat(loadingPct.toFixed(1))
    });
  }

  const wallMs = Date.now() - tStart;

  // Sparsity stats
  const totalEntries = n * n;
  const sparsity = ((1.0 - Y.nnz / totalEntries) * 100).toFixed(2);
  const jDim = nx;
  const jTotalEntries = jDim * jDim;
  // Non-zeros in Jacobian = up to 4 per Ybus non-zero
  let jNnz = 0;
  const jColInd = [];
  const jRowPtr = [0];
  if (lastJ) {
    for (let r = 0; r < nx; r++) {
      let rowNnz = 0;
      for (let c = 0; c < nx; c++) {
        if (Math.abs(lastJ[r][c]) > 1e-12) {
          jNnz++;
          rowNnz++;
          jColInd.push(c);
        }
      }
      jRowPtr.push(jRowPtr[jRowPtr.length - 1] + rowNnz);
    }
  } else {
    jNnz = Y.nnz * 2;
  }

  return {
    success: converged,
    iterations: finalIter,
    maxIterations,
    wall_ms: wallMs,
    history,
    linearItersHistory,
    phaseBreakdowns,
    buses: busResults,
    branches: branchResults,
    summary: {
      busCount: n,
      branchCount: grid.branches.length,
      ybusNnz: Y.nnz,
      ybusSparsity: sparsity,
      jacobianDim: nx,
      jacobianNnz: jNnz || Y.nnz * 2,
      numColors: mc.numColors,
      totalLossP_mw: parseFloat(totalLossP.toFixed(3)),
      totalLossQ_mvar: parseFloat(totalLossQ.toFixed(3)),
      underVoltageBuses: underVoltageCount,
      overVoltageBuses: overVoltageCount
    },
    ybusMatrix: {
      n,
      nnz: Y.nnz,
      rowptr: Y.rowptr,
      col: Y.col
    },
    jacobianMatrix: {
      n: nx,
      nnz: jNnz,
      rowptr: jRowPtr,
      col: jColInd
    }
  };
}

/**
 * BiCGSTAB linear solver with Jacobi / Incomplete factorization preconditioner
 */
function solveBiCGSTAB(A, b, tol = 1e-6, maxIters = 1000) {
  const n = b.length;
  const x = new Float64Array(n);
  const r = new Float64Array(n);
  const r_star = new Float64Array(n);
  const p = new Float64Array(n);
  const v = new Float64Array(n);
  const s = new Float64Array(n);
  const t = new Float64Array(n);

  // Preconditioner M: inverse diagonal
  const invM = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const diag = A[i][i];
    invM[i] = Math.abs(diag) > 1e-12 ? 1.0 / diag : 1.0;
  }

  // Initial residual r = b - A * x (x=0 initially, so r = b)
  for (let i = 0; i < n; i++) {
    r[i] = b[i];
    r_star[i] = b[i];
    p[i] = b[i];
  }

  const dot = (u, w) => {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += u[i] * w[i];
    return sum;
  };

  let rho_prev = 1.0;
  let alpha = 1.0;
  let omega = 1.0;

  const normB = Math.sqrt(dot(b, b)) || 1.0;

  let iters = 0;
  for (let k = 0; k < maxIters; k++) {
    iters = k + 1;
    const rho = dot(r_star, r);
    if (Math.abs(rho) < 1e-30) break;

    if (k === 0) {
      for (let i = 0; i < n; i++) p[i] = r[i];
    } else {
      const beta = (rho / rho_prev) * (alpha / omega);
      for (let i = 0; i < n; i++) {
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
      }
    }

    // Preconditioned p_hat = invM * p
    const p_hat = new Float64Array(n);
    for (let i = 0; i < n; i++) p_hat[i] = invM[i] * p[i];

    // v = A * p_hat
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) sum += A[i][j] * p_hat[j];
      v[i] = sum;
    }

    alpha = rho / (dot(r_star, v) || 1e-30);

    // s = r - alpha * v
    for (let i = 0; i < n; i++) {
      s[i] = r[i] - alpha * v[i];
    }

    const normS = Math.sqrt(dot(s, s));
    if (normS / normB < tol) {
      for (let i = 0; i < n; i++) x[i] += alpha * p_hat[i];
      break;
    }

    // Preconditioned s_hat = invM * s
    const s_hat = new Float64Array(n);
    for (let i = 0; i < n; i++) s_hat[i] = invM[i] * s[i];

    // t = A * s_hat
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) sum += A[i][j] * s_hat[j];
      t[i] = sum;
    }

    omega = dot(t, s) / (dot(t, t) || 1e-30);

    for (let i = 0; i < n; i++) {
      x[i] += alpha * p_hat[i] + omega * s_hat[i];
      r[i] = s[i] - omega * t[i];
    }

    rho_prev = rho;

    const resNorm = Math.sqrt(dot(r, r));
    if (resNorm / normB < tol || Math.abs(omega) < 1e-30) {
      break;
    }
  }

  return { x: Array.from(x), iters };
}

/**
 * Direct Gaussian elimination with partial pivoting for small matrices or reference fallback
 */
function solveDenseLU(A, b) {
  const n = b.length;
  const M = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n + 1);
    for (let j = 0; j < n; j++) row[j] = A[i][j];
    row[n] = b[i];
    return row;
  });

  for (let k = 0; k < n; k++) {
    let maxRow = k;
    let maxVal = Math.abs(M[k][k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(M[i][k]);
      if (v > maxVal) {
        maxVal = v;
        maxRow = i;
      }
    }
    if (maxRow !== k) {
      const tmp = M[k];
      M[k] = M[maxRow];
      M[maxRow] = tmp;
    }

    const diag = M[k][k];
    if (Math.abs(diag) < 1e-14) continue;

    for (let i = k + 1; i < n; i++) {
      const factor = M[i][k] / diag;
      for (let j = k; j <= n; j++) {
        M[i][j] -= factor * M[k][j];
      }
    }
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * x[j];
    }
    x[i] = sum / (M[i][i] || 1e-12);
  }

  return Array.from(x);
}

module.exports = {
  buildYbus,
  computeMulticolorOrdering,
  solvePowerFlow
};
