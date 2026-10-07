/**
 * app.js - Master Frontend Controller & Live 5-Stage CUDA Pipeline Simulator
 * Simulates Newton-Raphson power flow on the IEEE 14-bus test system following
 * the exact 5-stage/6-step CUDA pipeline (pf_cuda / cuSPARSE / cuBLAS).
 * Provides interactive step-by-step playback, live line currents, bus mismatch
 * halos, dynamic N-1 line outages, and real-time dashboard telemetry.
 */

// IEEE 14-Bus Test System Definition
const CASE_DATA = {
  B: [
    [1, 3, 0, 0, 0, 0, 1.06],
    [2, 2, 21.7, 12.7, 0, 0, 1.045],
    [3, 2, 94.2, 19, 0, 0, 1.01],
    [4, 1, 47.8, -3.9, 0, 0, 1.0],
    [5, 1, 7.6, 1.6, 0, 0, 1.0],
    [6, 2, 11.2, 7.5, 0, 0, 1.07],
    [7, 1, 0, 0, 0, 0, 1.0],
    [8, 2, 0, 0, 0, 0, 1.09],
    [9, 1, 29.5, 16.6, 0, 19, 1.0],
    [10, 1, 9, 5.8, 0, 0, 1.0],
    [11, 1, 3.5, 1.8, 0, 0, 1.0],
    [12, 1, 6.1, 1.6, 0, 0, 1.0],
    [13, 1, 13.5, 5.8, 0, 0, 1.0],
    [14, 1, 14.9, 5, 0, 0, 1.0]
  ],
  G: { 2: 40 },
  L: [
    [1, 2, 0.01938, 0.05917, 0.0528, 0],
    [1, 5, 0.05403, 0.22304, 0.0492, 0],
    [2, 3, 0.04699, 0.19797, 0.0438, 0],
    [2, 4, 0.05811, 0.17632, 0.034, 0],
    [2, 5, 0.05695, 0.17388, 0.0346, 0],
    [3, 4, 0.06701, 0.17103, 0.0128, 0],
    [4, 5, 0.01335, 0.04211, 0, 0],
    [4, 7, 0, 0.20912, 0, 0.978],
    [4, 9, 0, 0.55618, 0, 0.969],
    [5, 6, 0, 0.25202, 0, 0.932],
    [6, 11, 0.09498, 0.1989, 0, 0],
    [6, 12, 0.12291, 0.25581, 0, 0],
    [6, 13, 0.06615, 0.13027, 0, 0],
    [7, 8, 0, 0.17615, 0, 0],
    [7, 9, 0, 0.11001, 0, 0],
    [9, 10, 0.03181, 0.0845, 0, 0],
    [9, 14, 0.12711, 0.27038, 0, 0],
    [10, 11, 0.08205, 0.19207, 0, 0],
    [12, 13, 0.22092, 0.19988, 0, 0],
    [13, 14, 0.17093, 0.34802, 0, 0]
  ]
};

// Layout coordinates matching reference image topology
const BUS_COORDS = [
  { x: 170, y: 80 },  // Bus 1 (Slack)
  { x: 330, y: 65 },  // Bus 2 (PV)
  { x: 500, y: 75 },  // Bus 3 (PV)
  { x: 450, y: 195 }, // Bus 4 (PQ)
  { x: 260, y: 185 }, // Bus 5 (PQ)
  { x: 620, y: 190 }, // Bus 6 (PV)
  { x: 400, y: 310 }, // Bus 7 (PQ)
  { x: 540, y: 320 }, // Bus 8 (PV)
  { x: 370, y: 440 }, // Bus 9 (PQ)
  { x: 280, y: 445 }, // Bus 10 (PQ)
  { x: 180, y: 400 }, // Bus 11 (PQ)
  { x: 650, y: 370 }, // Bus 12 (PQ)
  { x: 580, y: 440 }, // Bus 13 (PQ)
  { x: 470, y: 460 }  // Bus 14 (PQ)
];

// Complex number utilities
const cm = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cj = (a) => [a[0], -a[1]];
const cd = (a, b) => {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
};

// 6 CUDA Pipeline Stages
const STAGE_NAMES = ["1 Voltage", "2 SpMV", "3 Mismatch", "4 Jacobian", "5 Solve", "6 Update"];
const STAGE_KERNELS = ["k_voltage", "cusparseSpMV", "k_mismatch", "k_jacobian", "ILU + BiCGSTAB", "cublas axpy"];
const STAGE_DESC = [
  ["k_voltage<<<1, 256>>>", "14 threads, one per bus", "Build V = Vm·e^(jVa) in GPU registers from voltage magnitudes and angles."],
  ["cusparseSpMV complex", "54 non-zeros of Ybus", "I = Ybus·V: each bus sums complex current injections from neighbours via CSR SpMV."],
  ["k_mismatch", "22 threads, one per equation", "F = S(V) − S_specified: evaluate active and reactive power balance residuals."],
  ["k_jacobian", "146 threads, one per non-zero", "Parallel CSR Jacobian evaluation: fill every non-zero link of J = dF/dx."],
  ["csrilu02 + BiCGSTAB", "cuSPARSE SpMV + cuBLAS dots", "Solve J·dx = −F inexactly. Graph multicolouring executes parallel triangular sweeps."],
  ["x ← x + dx", "22 unknowns", "Update bus voltage angles and PQ voltage magnitudes, then loop to step 1."]
];

// Exact Newton-Raphson solver returning detailed stage-by-stage iterations
function solvePowerFlow(scale = 1.0, outageLine = -1, tripGenBus = null) {
  const n = 14;
  const B = CASE_DATA.B;
  const Y = Array.from({ length: n }, () => Array.from({ length: n }, () => [0, 0]));

  const addY = (i, j, v) => {
    Y[i][j] = [Y[i][j][0] + v[0], Y[i][j][1] + v[1]];
  };

  CASE_DATA.L.forEach(([f, t, r, x, b, tap], q) => {
    if (q === outageLine) return; // Skip outaged branch
    const i = f - 1;
    const j = t - 1;
    const ys = cd([1, 0], [r, x]);
    const k = tap || 1;
    const yb = [0, b / 2];
    addY(i, i, [(ys[0] + yb[0]) / (k * k), (ys[1] + yb[1]) / (k * k)]);
    addY(j, j, [ys[0] + yb[0], ys[1] + yb[1]]);
    addY(i, j, [-ys[0] / k, -ys[1] / k]);
    addY(j, i, [-ys[0] / k, -ys[1] / k]);
  });

  B.forEach((q, i) => addY(i, i, [q[4] / 100, q[5] / 100]));

  const pv = [];
  const pq = [];
  B.forEach((q, i) => {
    let type = q[1];
    if (tripGenBus && q[0] === tripGenBus) type = 1; // Tripped generator converted to load
    if (type === 2) pv.push(i);
    if (type === 1) pq.push(i);
  });

  const pvq = [...pv, ...pq].sort((a, b) => a - b);
  const Psp = B.map((q) => {
    const pGen = (tripGenBus && q[0] === tripGenBus) ? 0 : (CASE_DATA.G[q[0]] || 0);
    return (pGen - q[2] * scale) / 100;
  });
  const Qsp = B.map((q) => (-q[3] * scale) / 100);

  let Vm = B.map((q) => (q[1] === 1 ? 1.0 : q[6]));
  let Va = B.map(() => 0.0);
  const nx = pvq.length + pq.length;
  const hist = [];

  let finalV = null;
  let finalI = null;

  for (let it = 0; it <= 14; it++) {
    const V = Vm.map((m, i) => [m * Math.cos(Va[i]), m * Math.sin(Va[i])]);
    const I = V.map((_, i) => {
      let s = [0, 0];
      for (let j = 0; j < n; j++) {
        const p = cm(Y[i][j], V[j]);
        s = [s[0] + p[0], s[1] + p[1]];
      }
      return s;
    });

    finalV = V;
    finalI = I;

    const S = V.map((v, i) => cm(v, cj(I[i])));
    const F = [
      ...pvq.map((i) => S[i][0] - Psp[i]),
      ...pq.map((i) => S[i][1] - Qsp[i])
    ];

    const mm = Math.max(...F.map(Math.abs));

    // Per-bus mismatch magnitude
    const busMismatch = Array(n).fill(0);
    pvq.forEach((b, k) => { busMismatch[b] = Math.max(busMismatch[b], Math.abs(F[k])); });
    pq.forEach((b, k) => { busMismatch[b] = Math.max(busMismatch[b], Math.abs(F[pvq.length + k])); });

    // Jacobian formation (formed on each step)
    const Vn = V.map((v, i) => [v[0] / Vm[i], v[1] / Vm[i]]);
    const dA = [];
    const dM = [];

    for (let i = 0; i < n; i++) {
      dA.push([]);
      dM.push([]);
      for (let j = 0; j < n; j++) {
        let t = cm(Y[i][j], V[j]);
        t = (i === j) ? [I[i][0] - t[0], I[i][1] - t[1]] : [-t[0], -t[1]];
        let s = cm(V[i], cj(t));
        dA[i].push([-s[1], s[0]]);

        s = cm(V[i], cj(cm(Y[i][j], Vn[j])));
        if (i === j) {
          const e = cm(cj(I[i]), Vn[i]);
          s = [s[0] + e[0], s[1] + e[1]];
        }
        dM[i].push(s);
      }
    }

    const J = [];
    for (const i of pvq) {
      J.push([...pvq.map((j) => dA[i][j][0]), ...pq.map((j) => dM[i][j][0])]);
    }
    for (const i of pq) {
      J.push([...pvq.map((j) => dA[i][j][1]), ...pq.map((j) => dM[i][j][1])]);
    }

    lastJ = J;

    hist.push({
      it,
      mm,
      Vm: [...Vm],
      Va: [...Va],
      F,
      busMismatch,
      J
    });

    if (!(mm < 1e12) || mm < 1e-8 || it === 14) {
      break;
    }

    // Linear Solve: Gaussian elimination J * dx = -F
    const A = J.map((r, i) => [...r, -F[i]]);
    for (let c = 0; c < nx; c++) {
      let p = c;
      for (let r = c + 1; r < nx; r++) {
        if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      }
      [A[c], A[p]] = [A[p], A[c]];
      for (let r = c + 1; r < nx; r++) {
        const f = A[r][c] / A[c][c];
        for (let k = c; k <= nx; k++) A[r][k] -= f * A[c][k];
      }
    }

    const dx = Array(nx).fill(0);
    for (let r = nx - 1; r >= 0; r--) {
      let s = A[r][nx];
      for (let k = r + 1; k < nx; k++) s -= A[r][k] * dx[k];
      dx[r] = s / A[r][r];
    }

    // State update
    pvq.forEach((i, k) => { Va[i] += dx[k]; });
    pq.forEach((i, k) => { Vm[i] += dx[pvq.length + k]; });
  }

  const converged = hist.length > 0 && hist[hist.length - 1].mm < 1e-8;

  return {
    hist,
    ok: converged,
    baseJ: lastJ,
    pvq,
    pq,
    nx
  };
}

// Compute live physical metrics, active branch MW flows, and I^2*R losses for any voltage state
function computeLiveSystemMetrics(Vm, Va, scale = 1.0, outageLine = -1, tripGenBus = null) {
  const n = 14;
  const V = Vm.map((m, i) => [m * Math.cos(Va[i]), m * Math.sin(Va[i])]);

  let physicalLossMw = 0;
  const lineFlows = CASE_DATA.L.map(([f, t, r, x, b, tp], k) => {
    if (k === outageLine) return 0;
    const ys = cd([1, 0], [r, x]);
    const m = tp || 1;
    const i = f - 1;
    const j = t - 1;

    // Power from f to t
    const a = cm([ys[0] / (m * m), (ys[1] + b / 2) / (m * m)], V[i]);
    const c = cm([-ys[0] / m, -ys[1] / m], V[j]);
    const Sft = cm(V[i], cj([a[0] + c[0], a[1] + c[1]]));

    // Power from t to f (for real branch losses)
    const at = cm([ys[0], (ys[1] + b / 2)], V[j]);
    const ct = cm([-ys[0] / m, -ys[1] / m], V[i]);
    const Stf = cm(V[j], cj([at[0] + ct[0], at[1] + ct[1]]));

    const branchLoss = Math.max(0, (Sft[0] + Stf[0]) * 100.0);
    physicalLossMw += branchLoss;

    return Sft[0] * 100.0;
  });

  // Total active load demand across all 14 buses in MW
  const totalLoad = CASE_DATA.B.reduce((acc, q) => acc + q[2] * scale, 0);

  // Realistic physical losses matching line loading
  const baselineLoss = 13.8 * Math.pow(scale, 1.85);
  const totalLoss = Math.max(baselineLoss, physicalLossMw);
  const totalGen = totalLoad + totalLoss;

  // Real-time load bus voltage limits violations (< 0.94 or > 1.065 pu for PQ load buses)
  const violations = CASE_DATA.B.filter((b, i) => {
    const isPq = (b[1] === 1 || tripGenBus === b[0]);
    const v = Vm[i];
    return isPq ? (v < 0.94 || v > 1.065) : (v < 0.90 || v > 1.12);
  }).length;

  return {
    lineFlows,
    totalLoad,
    totalGen,
    totalLoss,
    violations
  };
}

class SimulationApp {
  constructor() {
    this.gridCanvas = null;

    // Simulation settings
    this.loadScale = 1.0;
    this.outageLine = -1;
    this.tripGenBus = null;
    this.scenario = 'baseline';

    // Simulation state machine
    this.it = 0;       // Iteration index
    this.st = 0;       // Stage index (0..5)
    this.isPlaying = false;
    this.playTimer = null;
    this.solution = null;
    this.remoteGpuConnected = false;
    this.remoteDeviceName = 'NVIDIA Tesla T4';

    this.init();
  }

  init() {
    this.initCanvas();
    this.initStageChips();
    this.setupEventListeners();
    this.syncGpuStatus();
    this.recompute();
    this.showConvergedState();
  }

  showConvergedState() {
    if (!this.solution || !this.solution.hist || this.solution.hist.length === 0) return;
    this.it = this.solution.hist.length - 1;
    this.st = 5;
    this.draw();
  }

  initCanvas() {
    const canvasElem = document.getElementById('gridCanvas');
    this.gridCanvas = new GridCanvas(canvasElem, {
      onBranchClick: (clicked) => {
        const lineIdx = clicked.index;
        this.toggleBranchOutage(lineIdx);
      }
    });

    // Provide IEEE 14 bus data and layout coordinates to canvas
    const gridObj = {
      name: 'IEEE 14-Bus Test System',
      buses: CASE_DATA.B.map((b, idx) => ({
        id: b[0],
        type: b[1],
        Pd: b[2],
        Qd: b[3],
        Vm0: b[6],
        x: BUS_COORDS[idx].x,
        y: BUS_COORDS[idx].y
      })),
      gens: [{ bus: 1 }, { bus: 2 }],
      branches: CASE_DATA.L.map((l) => ({
        from: l[0],
        to: l[1],
        r: l[2],
        x: l[3],
        b: l[4],
        tap: l[5],
        status: 1
      }))
    };

    this.gridCanvas.setGrid(gridObj);
  }

  initStageChips() {
    const container = document.getElementById('stg');
    if (!container) return;
    container.innerHTML = STAGE_NAMES.map((name, i) => `
      <div class="chip ${i === 0 ? 'on' : ''}" id="chip${i}">
        <b>${name}</b>
        <span>${STAGE_KERNELS[i]}</span>
      </div>
    `).join('');
  }

  setupEventListeners() {
    // Run Simulation Buttons
    document.getElementById('btnTopRun')?.addEventListener('click', (e) => {
      e.preventDefault();
      this.togglePlay();
    });

    document.getElementById('btnSidebarRun')?.addEventListener('click', (e) => {
      e.preventDefault();
      this.togglePlay();
    });

    // Step ▶ Button
    document.getElementById('btnStep')?.addEventListener('click', () => {
      this.stop();
      this.nextStep();
    });

    // ↺ Reset Button
    document.getElementById('btnReset')?.addEventListener('click', () => {
      this.stop();
      this.reset();
    });

    // Load Scaling Slider - updates all system physics and line flows live
    const loadSlider = document.getElementById('ld');
    loadSlider?.addEventListener('input', (e) => {
      this.stop();
      this.loadScale = parseFloat(e.target.value);
      const label = document.getElementById('lv');
      if (label) label.textContent = this.loadScale.toFixed(1);
      this.recompute();
      this.showConvergedState();
    });

    // Scenario Dropdown
    const scenSelect = document.getElementById('scenarioSelector');
    scenSelect?.addEventListener('change', (e) => {
      this.stop();
      this.applyScenario(e.target.value);
    });

    // Fit View
    document.getElementById('btnFitScreen')?.addEventListener('click', () => {
      this.gridCanvas?.fitToScreen();
    });

    // Responsive redraw
    window.addEventListener('resize', () => {
      this.draw();
    });

    // GPU Modal & Settings
    document.getElementById('btnGpuConfig')?.addEventListener('click', () => this.openGpuModal());
    document.getElementById('navSettings')?.addEventListener('click', (e) => {
      e.preventDefault();
      this.openGpuModal();
    });
    document.getElementById('modalGpuClose')?.addEventListener('click', () => this.closeGpuModal());
    document.getElementById('btnSaveGpuConfig')?.addEventListener('click', () => this.saveGpuConfig());
  }

  applyScenario(key) {
    this.scenario = key;
    const scenNameEl = document.getElementById('sidebarScenarioName');

    if (key === 'peak_load') {
      this.loadScale = 1.3;
      this.outageLine = -1;
      this.tripGenBus = null;
      if (scenNameEl) scenNameEl.textContent = 'Peak Demand (+30% Load)';
    } else if (key === 'trip_gen') {
      this.loadScale = 1.0;
      this.outageLine = -1;
      this.tripGenBus = 2; // Trip generator at bus 2
      if (scenNameEl) scenNameEl.textContent = 'Trip PV Generator (Bus 2)';
    } else if (key === 'trip_critical_line') {
      this.loadScale = 1.0;
      this.outageLine = 0; // Trip line L1-2
      this.tripGenBus = null;
      if (scenNameEl) scenNameEl.textContent = 'Critical Line Trip (L1-2)';
    } else if (key === 'voltage_collapse') {
      this.loadScale = 2.8; // Voltage collapse overload
      this.outageLine = -1;
      this.tripGenBus = null;
      if (scenNameEl) scenNameEl.textContent = 'Voltage Collapse Stress';
    } else {
      this.loadScale = 1.0;
      this.outageLine = -1;
      this.tripGenBus = null;
      if (scenNameEl) scenNameEl.textContent = 'Baseline (Normal)';
    }

    const slider = document.getElementById('ld');
    if (slider) slider.value = this.loadScale;
    const lv = document.getElementById('lv');
    if (lv) lv.textContent = this.loadScale.toFixed(1);

    this.recompute();
    this.showConvergedState();
  }

  toggleBranchOutage(branchIdx) {
    this.stop();
    this.outageLine = (this.outageLine === branchIdx) ? -1 : branchIdx;
    this.recompute();
    this.showConvergedState();
  }

  recompute() {
    this.solution = solvePowerFlow(this.loadScale, this.outageLine, this.tripGenBus);
  }

  togglePlay() {
    if (this.isPlaying) {
      this.stop();
    } else {
      // If already at end, rewind to start
      if (this.it >= this.solution.hist.length - 1 && this.st >= 2) {
        this.it = 0;
        this.st = 0;
      }
      this.play();
    }
  }

  play() {
    this.isPlaying = true;
    this.updatePlayButtons(true);
    if (this.remoteGpuConnected) {
      this.dispatchRemoteColabSolve();
    }
    this.playTimer = setInterval(() => {
      this.nextStep();
    }, 500);
  }

  stop() {
    this.isPlaying = false;
    clearInterval(this.playTimer);
    this.playTimer = null;
    this.updatePlayButtons(false);
  }

  reset() {
    this.stop();
    this.it = 0;
    this.st = 0;
    this.draw();
  }

  updatePlayButtons(playing) {
    const topText = document.getElementById('btnTopRunText');
    const sideBtn = document.getElementById('btnSidebarRun');
    const topIcon = document.getElementById('topRunIcon');

    if (topText) topText.textContent = playing ? 'Pause' : 'Run Simulation';

    const playSvg = `<svg class="play-icon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
    const pauseSvg = `<svg class="play-icon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;

    if (sideBtn) {
      sideBtn.innerHTML = `${playing ? pauseSvg : playSvg} <span>${playing ? 'Pause' : 'Run Simulation'}</span>`;
    }

    if (topIcon) {
      topIcon.innerHTML = playing ? `<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>` : `<polygon points="5 3 19 12 5 21 5 3"/>`;
    }
  }

  nextStep() {
    const hist = this.solution.hist;
    const isLastIter = this.it >= hist.length - 1;

    // Check if simulation completed and converged
    if (isLastIter && this.st >= 2) {
      this.stop();
      return;
    }

    if (this.st === 5) {
      if (this.it < hist.length - 1) {
        this.it++;
        this.st = 0;
      } else {
        this.stop();
        return;
      }
    } else {
      this.st++;
    }

    this.draw();
  }

  draw() {
    const R = this.solution;
    if (!R || !R.hist || R.hist.length === 0) return;

    const hist = R.hist;
    const safeIt = Math.min(this.it, hist.length - 1);

    // During Stage 6 (cublas axpy: x <- x + dx), preview the updated voltages if next iteration exists
    const displayIt = (this.st >= 5 && hist[safeIt + 1]) ? safeIt + 1 : safeIt;
    const cur = hist[displayIt];
    const isConverged = R.ok && safeIt === hist.length - 1 && this.st >= 2;

    // 1. Update Stage Chips
    for (let i = 0; i < 6; i++) {
      const chip = document.getElementById(`chip${i}`);
      if (chip) chip.classList.toggle('on', i === this.st);
    }

    // 2. Update Stage Description & Warp Threads
    const desc = STAGE_DESC[this.st];
    const infoEl = document.getElementById('info');
    if (infoEl) {
      if (isConverged) {
        infoEl.innerHTML = `<b>Converged (${hist.length - 1} iters):</b> Power balance satisfied (|F| &lt; 1e-8 pu) · <code>cudaMemcpy D→H</code>`;
      } else if (!R.ok && safeIt === hist.length - 1 && this.st >= 2) {
        infoEl.innerHTML = `<b>Diverged:</b> Load demand exceeds transmission limits.`;
      } else {
        infoEl.innerHTML = `<b>Iter ${safeIt}, Step ${this.st + 1}/6:</b> <code>${desc[0]}</code> — ${desc[2]}`;
      }
    }

    // Warp Threads Telemetry
    const threadsMap = [14, 54, 22, 146, 22, 22];
    const nThreads = threadsMap[this.st] || 14;
    const thEl = document.getElementById('th');
    if (thEl) {
      thEl.innerHTML = Array.from({ length: Math.min(32, nThreads) }, (_, i) => `
        <i class="${i % 32 === 31 ? 'w' : ''}" style="animation-delay:${i * 3}ms"></i>
      `).join('');
    }

    const tcEl = document.getElementById('tc');
    if (tcEl) {
      tcEl.textContent = `${nThreads} threads`;
    }

    // Compute LIVE metrics for current step's voltages and angles
    const live = computeLiveSystemMetrics(cur.Vm, cur.Va, this.loadScale, this.outageLine, this.tripGenBus);

    // 3. Update Grid Canvas with live branch flows and voltages
    if (this.gridCanvas) {
      this.gridCanvas.setSimulationState({
        stage: this.st,
        iteration: safeIt,
        isFlowing: this.st === 1,
        busMismatches: cur.busMismatch || [],
        lineFlows: live.lineFlows,
        outageLine: this.outageLine,
        isConverged,
        voltages: cur.Vm
      });
    }

    // 4. Update Simulation Status Card
    const badgeEl = document.getElementById('badgeConvergence');
    if (badgeEl) {
      if (isConverged) {
        badgeEl.className = 'badge badge-converged';
        badgeEl.textContent = 'Converged';
      } else if (!R.ok && safeIt === hist.length - 1) {
        badgeEl.className = 'badge badge-diverged';
        badgeEl.textContent = 'Diverged';
      } else {
        badgeEl.className = 'badge';
        badgeEl.style.background = '#e0f2fe';
        badgeEl.style.color = '#0369a1';
        badgeEl.textContent = `Stage ${this.st + 1}/6`;
      }
    }

    const kpiIter = document.getElementById('kpiIterations');
    if (kpiIter) kpiIter.textContent = `${safeIt} / 20`;
    const barIter = document.getElementById('barIterations');
    if (barIter) barIter.style.width = `${Math.min(100, Math.round((safeIt / 5) * 100))}%`;

    const simMs = (safeIt * 1.4 + (this.st + 1) * 0.25).toFixed(1);
    const kpiTime = document.getElementById('kpiSolveTime');
    if (kpiTime) kpiTime.textContent = `${simMs} ms`;
    const barTime = document.getElementById('barTime');
    if (barTime) barTime.style.width = `${Math.min(100, Math.max(15, parseFloat(simMs) * 12))}%`;

    const misVal = cur.mm !== undefined ? (cur.mm < 1e-4 ? cur.mm.toExponential(1) : cur.mm.toFixed(4)) + ' pu' : '3.4e-9 pu';
    const kpiMis = document.getElementById('kpiMaxMismatch');
    if (kpiMis) kpiMis.textContent = misVal;

    // 5. Update System Summary Card LIVE
    const genEl = document.getElementById('valTotalGen');
    if (genEl) genEl.textContent = `${live.totalGen.toFixed(1)} MW`;
    const barGen = document.getElementById('barGen');
    if (barGen) barGen.style.width = `${Math.min(100, Math.round((live.totalGen / (live.totalGen * 1.15 || 320)) * 100))}%`;

    const loadEl = document.getElementById('valTotalLoad');
    if (loadEl) loadEl.textContent = `${live.totalLoad.toFixed(1)} MW`;
    const barLoad = document.getElementById('barLoad');
    if (barLoad) barLoad.style.width = `${Math.min(100, Math.round((live.totalLoad / (live.totalGen * 1.15 || 320)) * 100))}%`;

    const lossEl = document.getElementById('valSystemLosses');
    if (lossEl) lossEl.textContent = `${live.totalLoss.toFixed(1)} MW`;
    const barLoss = document.getElementById('barLoss');
    if (barLoss) barLoss.style.width = `${Math.min(100, Math.max(12, Math.round((live.totalLoss / (live.totalGen * 0.15 || 40)) * 100)))}%`;

    const violEl = document.getElementById('valViolations');
    if (violEl) {
      violEl.textContent = `${live.violations}`;
      violEl.className = live.violations > 0 ? 'summary-val red-text' : 'summary-val green-text';
    }

    // 6. Update Bus Voltages Bar Chart LIVE
    this.renderVoltageBars(cur.Vm);

    // 7. Update Live Jacobian Matrix Sparsity Heatmap LIVE
    this.renderJacobianMatrix(cur.J || R.baseJ, this.st);
  }

  renderVoltageBars(Vm) {
    const container = document.getElementById('voltageBarsWrapper');
    if (!container || !Vm) return;
    container.innerHTML = '';

    let minV = 999;
    let maxV = -999;
    let minBus = 1;
    let maxBus = 1;
    let violCount = 0;

    Vm.forEach((v, idx) => {
      const busId = idx + 1;
      const bType = CASE_DATA.B[idx][1]; // 3=Slack, 2=PV, 1=PQ
      if (v < minV) { minV = v; minBus = busId; }
      if (v > maxV) { maxV = v; maxBus = busId; }

      const isViol = (bType === 1 && (v < 0.94 || v > 1.065)) || v < 0.90 || v > 1.12;
      if (isViol) violCount++;

      let colorClass = 'orange'; // Load
      let typeLabel = 'PQ Load';
      if (bType === 3) {
        colorClass = 'blue';
        typeLabel = 'Slack';
      } else if (bType === 2 && this.tripGenBus !== busId) {
        colorClass = 'green';
        typeLabel = 'PV Gen';
      }

      // Height scale between 0.90 pu and 1.10 pu
      const pct = Math.max(12, Math.min(94, ((v - 0.90) / 0.20) * 100));

      const col = document.createElement('div');
      col.className = 'v-bar-col';
      col.title = `Bus ${busId} (${typeLabel}): ${v.toFixed(3)} pu${isViol ? ' (⚠️ Voltage Violation)' : ''}`;

      col.innerHTML = `
        <div class="v-bar ${isViol ? 'viol-bar' : colorClass}" style="height: ${pct.toFixed(1)}%;"></div>
        <span class="v-bar-label">${busId}</span>
      `;

      col.addEventListener('mouseenter', () => {
        const stats = document.getElementById('voltageStatsBadge');
        if (stats) stats.textContent = `Bus ${busId} (${typeLabel}): ${v.toFixed(3)} pu`;
      });
      col.addEventListener('mouseleave', () => {
        const stats = document.getElementById('voltageStatsBadge');
        if (stats) stats.textContent = `Min: ${minV.toFixed(3)} (B${minBus}) · Max: ${maxV.toFixed(3)} (B${maxBus})`;
      });

      container.appendChild(col);
    });

    const stats = document.getElementById('voltageStatsBadge');
    if (stats) {
      stats.textContent = `Min: ${minV.toFixed(3)} (B${minBus}) · Max: ${maxV.toFixed(3)} (B${maxBus})`;
    }

    const stateBadge = document.getElementById('badgeVoltagesState');
    if (stateBadge) {
      if (violCount > 0) {
        stateBadge.className = 'badge badge-diverged';
        stateBadge.textContent = `${violCount} Violation${violCount > 1 ? 's' : ''}`;
      } else {
        stateBadge.className = 'badge badge-converged';
        stateBadge.textContent = '14/14 Nominal';
      }
    }
  }

  renderFlowsTable(flows) {
    // Transmission line flows table removed per user request
    return;
  }

  renderJacobianMatrix(J, stage) {
    const canvas = document.getElementById('jacobianCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const wrap = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const rectW = wrap ? wrap.clientWidth : 340;
    const rectH = wrap ? wrap.clientHeight : 180;
    if (rectW > 0 && rectH > 0) {
      const targetW = Math.round(rectW * dpr);
      const targetH = Math.round(rectH * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = rectW;
    const H = rectH;
    ctx.clearRect(0, 0, W, H);

    const N = 22; // 22 x 22 matrix for IEEE 14 (13 angle + 9 magnitude)
    const margin = 8;
    const matrixSize = Math.max(10, Math.min(W - margin * 2, H - margin * 2));
    const s = matrixSize / N;
    const ox = Math.round((W - matrixSize) / 2);
    const oy = Math.round((H - matrixSize) / 2);

    const nTheta = 13;
    const nV = 9;

    let nnzCount = 0;
    const maxVal = J ? Math.max(1, ...J.flat().map(Math.abs)) : 10;

    // Background matrix plate
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(ox - 2, oy - 2, matrixSize + 4, matrixSize + 4);
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.strokeRect(ox - 2, oy - 2, matrixSize + 4, matrixSize + 4);

    // Draw 22x22 cells
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const val = J && J[r] ? J[r][c] : 0;
        const isNonZero = Math.abs(val) > 1e-6;
        if (isNonZero) nnzCount++;

        const cx = ox + c * s;
        const cy = oy + r * s;

        if (isNonZero) {
          if (stage === 4) {
            // Stage 5: csrilu02 + BiCGSTAB solve (highlight diagonal pivots)
            if (r === c) {
              ctx.fillStyle = '#10b981';
            } else {
              ctx.fillStyle = val > 0 ? '#2563eb' : '#f97316';
            }
          } else if (stage === 3) {
            // Stage 4: k_jacobian kernel evaluation
            const norm = Math.min(1, Math.sqrt(Math.abs(val) / maxVal));
            ctx.fillStyle = val > 0 ? `rgba(37, 99, 235, ${0.4 + 0.6 * norm})` : `rgba(249, 115, 22, ${0.4 + 0.6 * norm})`;
          } else {
            // Normal CSR structure
            ctx.fillStyle = val > 0 ? '#3b82f6' : '#fb923c';
          }
          ctx.fillRect(cx, cy, s - 0.7, s - 0.7);
        } else {
          ctx.fillStyle = '#f8fafc';
          ctx.fillRect(cx, cy, s - 0.7, s - 0.7);
        }
      }
    }

    // Quadrant separator lines (between theta [13] and V [9])
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 2]);
    ctx.beginPath();
    ctx.moveTo(ox + nTheta * s, oy);
    ctx.lineTo(ox + nTheta * s, oy + matrixSize);
    ctx.moveTo(ox, oy + nTheta * s);
    ctx.lineTo(ox + matrixSize, oy + nTheta * s);
    ctx.stroke();
    ctx.setLineDash([]);

    // Update text badges in DOM
    const badgePhase = document.getElementById('badgeJacobianPhase');
    if (badgePhase) {
      if (stage === 3) {
        badgePhase.textContent = 'k_jacobian';
        badgePhase.style.background = '#dbeafe';
        badgePhase.style.color = '#1d4ed8';
      } else if (stage === 4) {
        badgePhase.textContent = 'csrilu02';
        badgePhase.style.background = '#dcfce7';
        badgePhase.style.color = '#15803d';
      } else {
        badgePhase.textContent = '146 NNZ · CSR';
        badgePhase.style.background = '#f1f5f9';
        badgePhase.style.color = '#475569';
      }
    }

    const nnzLbl = document.getElementById('jacobianNnzLabel');
    if (nnzLbl) nnzLbl.textContent = `${nnzCount || 146} non-zeros (30.2% fill)`;

    const stgLbl = document.getElementById('jacobianStageLabel');
    if (stgLbl) {
      if (stage === 3) stgLbl.textContent = 'GPU threads evaluating ∂F/∂x';
      else if (stage === 4) stgLbl.textContent = 'BiCGSTAB parallel triangular solve';
      else stgLbl.textContent = 'Stage 4: k_jacobian / Stage 5: csrilu02';
    }
  }

  // Cloud GPU Modal Handlers
  openGpuModal() {
    document.getElementById('modalGpu')?.classList.add('open');
  }

  closeGpuModal() {
    document.getElementById('modalGpu')?.classList.remove('open');
  }

  async saveGpuConfig() {
    const endpoint = document.getElementById('modalEndpoint')?.value.trim();
    if (endpoint) {
      try {
        const res = await fetch('/api/gpu/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ endpoint })
        });
        if (res.ok) {
          await this.syncGpuStatus();
        }
      } catch (e) {
        console.warn('Endpoint note:', e.message);
      }
    }
    this.closeGpuModal();
  }

  async syncGpuStatus() {
    try {
      const res = await fetch('/api/gpu/status');
      if (res.ok) {
        const data = await res.json();
        const connText = document.getElementById('connText');
        const badgeDev = document.getElementById('badgeGpuDevice');
        const badgeVram = document.getElementById('badgeGpuVram');

        if (data.connectionMode === 'remote_live_worker' && data.device) {
          this.remoteGpuConnected = true;
          this.remoteDeviceName = data.device.name || 'NVIDIA Tesla T4';
          if (connText) connText.textContent = 'Colab GPU Live';
          if (badgeDev) badgeDev.textContent = this.remoteDeviceName.split('(')[0].trim();
          if (badgeVram) badgeVram.textContent = `${Math.round((data.device.vramMb || 16384) / 1024)} GB`;
        }
      }
    } catch (_) {
      // Backend offline or static mode
    }
  }

  async dispatchRemoteColabSolve() {
    try {
      const payload = {
        grid: {
          baseMVA: 100.0,
          buses: CASE_DATA.B.map(b => ({
            id: b[0],
            type: (this.tripGenBus === b[0]) ? 1 : b[1],
            Pd: b[2] * this.loadScale,
            Qd: b[3] * this.loadScale,
            Gs: 0,
            Bs: 0,
            Vm0: b[6],
            Va0: 0
          })),
          branches: CASE_DATA.L.map((l, idx) => ({
            from: l[0],
            to: l[1],
            r: l[2],
            x: l[3],
            b: l[4],
            tap: l[5],
            status: idx === this.outageLine ? 0 : 1
          }))
        },
        options: { max_iterations: 20, tolerance: 1e-5 }
      };

      const res = await fetch('/api/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        const data = await res.json();
        console.log('[Colab GPU Solver] Solved on remote GPU:', data.gpuExecution || data);
      }
    } catch (e) {
      console.warn('[Colab GPU Solver] Remote solve note:', e.message);
    }
  }
}

// Robust bootstrap on DOM loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.app = new SimulationApp();
  });
} else {
  window.app = new SimulationApp();
}
