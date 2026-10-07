/**
 * cases.js - Power Grid Case Definitions, Parsers, and Synthetic Generators
 * Compatible with pf_cuda and pf_reference .case formats
 */

const fs = require('fs');
const path = require('path');

// Read a .case file from disk
function parseCaseFile(text, name = 'custom') {
  const lines = text.split('\n')
    .map(line => line.split('#')[0].trim())
    .filter(line => line.length > 0);

  let baseMVA = 100.0;
  const buses = [];
  const gens = [];
  const branches = [];

  let idx = 0;
  while (idx < lines.length) {
    const tokens = lines[idx].split(/\s+/);
    const key = tokens[0];

    if (key === 'baseMVA') {
      baseMVA = parseFloat(tokens[1]);
      idx++;
    } else if (key === 'buses') {
      const count = parseInt(tokens[1], 10);
      idx++;
      for (let k = 0; k < count && idx < lines.length; k++) {
        const parts = lines[idx].split(/\s+/).map(Number);
        buses.push({
          id: parts[0],
          type: parts[1], // 1=PQ, 2=PV, 3=Slack
          Pd: parts[2] || 0,
          Qd: parts[3] || 0,
          Gs: parts[4] || 0,
          Bs: parts[5] || 0,
          Vm0: parts[6] || 1.0,
          Va0: parts[7] || 0.0
        });
        idx++;
      }
    } else if (key === 'gens') {
      const count = parseInt(tokens[1], 10);
      idx++;
      for (let k = 0; k < count && idx < lines.length; k++) {
        const parts = lines[idx].split(/\s+/).map(Number);
        gens.push({
          bus: parts[0],
          Pg: parts[1] || 0,
          Vg: parts[2] || 1.0
        });
        idx++;
      }
    } else if (key === 'branches') {
      const count = parseInt(tokens[1], 10);
      idx++;
      for (let k = 0; k < count && idx < lines.length; k++) {
        const parts = lines[idx].split(/\s+/).map(Number);
        branches.push({
          from: parts[0],
          to: parts[1],
          r: parts[2] || 0.01,
          x: parts[3] || 0.05,
          b: parts[4] || 0,
          tap: parts[5] || 0, // 0 = standard line, >0 = transformer ratio
          status: 1 // 1 = in service, 0 = outaged
        });
        idx++;
      }
    } else {
      idx++;
    }
  }

  return {
    name,
    baseMVA,
    buses,
    gens,
    branches
  };
}

// Serialize a grid object to .case format
function serializeCaseFile(grid) {
  let out = `# Power Flow Case File: ${grid.name || 'Exported Grid'}\n`;
  out += `baseMVA ${grid.baseMVA || 100.0}\n`;
  out += `buses ${grid.buses.length}\n`;
  for (const b of grid.buses) {
    out += `${b.id} ${b.type} ${b.Pd} ${b.Qd} ${b.Gs || 0} ${b.Bs || 0} ${b.Vm0 || 1.0} ${b.Va0 || 0.0}\n`;
  }
  const gens = grid.gens || [];
  out += `gens ${gens.length}\n`;
  for (const g of gens) {
    out += `${g.bus} ${g.Pg} ${g.Vg}\n`;
  }
  const branches = (grid.branches || []).filter(br => br.status !== 0);
  out += `branches ${branches.length}\n`;
  for (const br of branches) {
    out += `${br.from} ${br.to} ${br.r} ${br.x} ${br.b} ${br.tap || 0}\n`;
  }
  return out;
}

// Layout coordinate presets for known IEEE test systems
const IEEE14_COORDS = {
  1: { x: 170, y: 90 },
  2: { x: 330, y: 70 },
  3: { x: 500, y: 80 },
  4: { x: 450, y: 210 },
  5: { x: 260, y: 200 },
  6: { x: 620, y: 200 },
  7: { x: 400, y: 320 },
  8: { x: 540, y: 340 },
  9: { x: 370, y: 460 },
  10: { x: 280, y: 465 },
  11: { x: 180, y: 420 },
  12: { x: 650, y: 390 },
  13: { x: 580, y: 460 },
  14: { x: 470, y: 480 }
};

// IEEE 14 Case loaded from cases/ieee14.case
function getIEEE14Case() {
  const casePath = path.join(__dirname, '../../cases/ieee14.case');
  if (fs.existsSync(casePath)) {
    const raw = fs.readFileSync(casePath, 'utf8');
    const parsed = parseCaseFile(raw, 'IEEE 14-Bus Test System');
    // Attach default coordinates
    parsed.buses.forEach(b => {
      if (IEEE14_COORDS[b.id]) {
        b.x = IEEE14_COORDS[b.id].x;
        b.y = IEEE14_COORDS[b.id].y;
      }
    });
    return parsed;
  }
  throw new Error('IEEE 14 case file not found');
}

// IEEE 30-Bus System (Standard MATPOWER case30)
function getIEEE30Case() {
  const baseMVA = 100.0;
  const buses = [
    { id: 1, type: 3, Pd: 0, Qd: 0, Gs: 0, Bs: 0, Vm0: 1.06, Va0: 0, x: 100, y: 120 },
    { id: 2, type: 2, Pd: 21.7, Qd: 12.7, Gs: 0, Bs: 0, Vm0: 1.043, Va0: -5.48, x: 200, y: 100 },
    { id: 3, type: 1, Pd: 2.4, Qd: 1.2, Gs: 0, Bs: 0, Vm0: 1.021, Va0: -7.96, x: 180, y: 220 },
    { id: 4, type: 1, Pd: 7.6, Qd: 1.6, Gs: 0, Bs: 0, Vm0: 1.012, Va0: -9.62, x: 260, y: 200 },
    { id: 5, type: 2, Pd: 94.2, Qd: 19.0, Gs: 0, Bs: 0, Vm0: 1.01, Va0: -14.37, x: 260, y: 320 },
    { id: 6, type: 1, Pd: 0, Qd: 0, Gs: 0, Bs: 0, Vm0: 1.01, Va0: -11.34, x: 380, y: 240 },
    { id: 7, type: 1, Pd: 22.8, Qd: 10.9, Gs: 0, Bs: 0, Vm0: 1.002, Va0: -13.12, x: 340, y: 360 },
    { id: 8, type: 2, Pd: 30.0, Qd: 30.0, Gs: 0, Bs: 0, Vm0: 1.01, Va0: -12.10, x: 460, y: 340 },
    { id: 9, type: 1, Pd: 0, Qd: 0, Gs: 0, Bs: 0, Vm0: 1.051, Va0: -14.38, x: 480, y: 200 },
    { id: 10, type: 1, Pd: 5.8, Qd: 2.0, Gs: 0, Bs: 19, Vm0: 1.045, Va0: -15.94, x: 420, y: 460 },
    { id: 11, type: 2, Pd: 0, Qd: 0, Gs: 0, Bs: 0, Vm0: 1.082, Va0: -14.39, x: 540, y: 160 },
    { id: 12, type: 1, Pd: 11.2, Qd: 7.5, Gs: 0, Bs: 0, Vm0: 1.057, Va0: -15.24, x: 580, y: 280 },
    { id: 13, type: 2, Pd: 0, Qd: 0, Gs: 0, Bs: 0, Vm0: 1.071, Va0: -15.24, x: 680, y: 260 },
    { id: 14, type: 1, Pd: 6.2, Qd: 1.6, Gs: 0, Bs: 0, Vm0: 1.042, Va0: -16.13, x: 600, y: 380 },
    { id: 15, type: 1, Pd: 8.2, Qd: 2.5, Gs: 0, Bs: 0, Vm0: 1.038, Va0: -16.22, x: 640, y: 440 },
    { id: 16, type: 1, Pd: 3.5, Qd: 1.8, Gs: 0, Bs: 0, Vm0: 1.045, Va0: -15.83, x: 540, y: 360 },
    { id: 17, type: 1, Pd: 9.0, Qd: 5.8, Gs: 0, Bs: 0, Vm0: 1.04, Va0: -16.14, x: 480, y: 400 },
    { id: 18, type: 1, Pd: 3.2, Qd: 0.9, Gs: 0, Bs: 0, Vm0: 1.028, Va0: -16.82, x: 700, y: 400 },
    { id: 19, type: 1, Pd: 9.5, Qd: 3.4, Gs: 0, Bs: 0, Vm0: 1.026, Va0: -17.00, x: 680, y: 480 },
    { id: 20, type: 1, Pd: 2.2, Qd: 0.7, Gs: 0, Bs: 0, Vm0: 1.03, Va0: -16.80, x: 600, y: 500 },
    { id: 21, type: 1, Pd: 17.5, Qd: 11.2, Gs: 0, Bs: 0, Vm0: 1.033, Va0: -16.42, x: 460, y: 520 },
    { id: 22, type: 1, Pd: 0, Qd: 0, Gs: 0, Bs: 0, Vm0: 1.033, Va0: -16.41, x: 420, y: 560 },
    { id: 23, type: 1, Pd: 3.2, Qd: 1.6, Gs: 0, Bs: 0, Vm0: 1.027, Va0: -16.61, x: 520, y: 560 },
    { id: 24, type: 1, Pd: 8.7, Qd: 6.7, Gs: 0, Bs: 4.3, Vm0: 1.021, Va0: -16.78, x: 380, y: 620 },
    { id: 25, type: 1, Pd: 0, Qd: 0, Gs: 0, Bs: 0, Vm0: 1.017, Va0: -16.35, x: 320, y: 580 },
    { id: 26, type: 1, Pd: 3.5, Qd: 2.3, Gs: 0, Bs: 0, Vm0: 1.0, Va0: -16.77, x: 260, y: 580 },
    { id: 27, type: 1, Pd: 0, Qd: 0, Gs: 0, Bs: 0, Vm0: 1.023, Va0: -15.82, x: 260, y: 480 },
    { id: 28, type: 1, Pd: 0, Qd: 0, Gs: 0, Bs: 0, Vm0: 1.007, Va0: -11.97, x: 320, y: 280 },
    { id: 29, type: 1, Pd: 2.4, Qd: 0.9, Gs: 0, Bs: 0, Vm0: 1.003, Va0: -17.06, x: 200, y: 540 },
    { id: 30, type: 1, Pd: 10.6, Qd: 1.9, Gs: 0, Bs: 0, Vm0: 0.992, Va0: -17.94, x: 140, y: 540 }
  ];

  const gens = [
    { bus: 1, Pg: 260.2, Vg: 1.06 },
    { bus: 2, Pg: 40.0, Vg: 1.043 },
    { bus: 5, Pg: 0.0, Vg: 1.01 },
    { bus: 8, Pg: 0.0, Vg: 1.01 },
    { bus: 11, Pg: 0.0, Vg: 1.082 },
    { bus: 13, Pg: 0.0, Vg: 1.071 }
  ];

  const rawBranches = [
    [1, 2, 0.0192, 0.0575, 0.0264, 0],
    [1, 3, 0.0452, 0.1652, 0.0204, 0],
    [2, 4, 0.0570, 0.1737, 0.0184, 0],
    [3, 4, 0.0132, 0.0379, 0.0042, 0],
    [2, 5, 0.0472, 0.1983, 0.0209, 0],
    [2, 6, 0.0581, 0.1763, 0.0187, 0],
    [4, 6, 0.0119, 0.0414, 0.0045, 0],
    [5, 7, 0.0460, 0.1160, 0.0102, 0],
    [6, 7, 0.0267, 0.0820, 0.0085, 0],
    [6, 8, 0.0120, 0.0420, 0.0045, 0],
    [6, 9, 0.0, 0.2080, 0.0, 0.978],
    [6, 10, 0.0, 0.5560, 0.0, 0.969],
    [9, 11, 0.0, 0.2080, 0.0, 0.0],
    [9, 10, 0.0, 0.1100, 0.0, 0.0],
    [4, 12, 0.0, 0.2560, 0.0, 0.932],
    [12, 13, 0.0, 0.1400, 0.0, 0.0],
    [12, 14, 0.1236, 0.2559, 0.0, 0],
    [12, 15, 0.0662, 0.1304, 0.0, 0],
    [12, 16, 0.0945, 0.1987, 0.0, 0],
    [14, 15, 0.2210, 0.1997, 0.0, 0],
    [16, 17, 0.0824, 0.1923, 0.0, 0],
    [15, 18, 0.1070, 0.2185, 0.0, 0],
    [18, 19, 0.0639, 0.1292, 0.0, 0],
    [19, 20, 0.0340, 0.0680, 0.0, 0],
    [10, 20, 0.0936, 0.2090, 0.0, 0],
    [10, 17, 0.0324, 0.0845, 0.0, 0],
    [10, 21, 0.0348, 0.0749, 0.0, 0],
    [10, 22, 0.0727, 0.1499, 0.0, 0],
    [21, 22, 0.0116, 0.0236, 0.0, 0],
    [15, 23, 0.1000, 0.2020, 0.0, 0],
    [22, 24, 0.1150, 0.1790, 0.0, 0],
    [23, 24, 0.1320, 0.2700, 0.0, 0],
    [24, 25, 0.1885, 0.3292, 0.0, 0],
    [25, 26, 0.2544, 0.3800, 0.0, 0],
    [25, 27, 0.1093, 0.2087, 0.0, 0],
    [28, 27, 0.0, 0.3960, 0.0, 0.968],
    [27, 29, 0.2198, 0.4153, 0.0, 0],
    [27, 30, 0.3202, 0.6027, 0.0, 0],
    [29, 30, 0.2399, 0.4533, 0.0, 0],
    [8, 28, 0.0636, 0.2000, 0.0214, 0],
    [6, 28, 0.0169, 0.0599, 0.065, 0]
  ];

  const branches = rawBranches.map(r => ({
    from: r[0],
    to: r[1],
    r: r[2],
    x: r[3],
    b: r[4],
    tap: r[5],
    status: 1
  }));

  return {
    name: 'IEEE 30-Bus System',
    baseMVA,
    buses,
    gens,
    branches
  };
}

// Pseudo-random generator matching powerflow.h Rng class
class Rng {
  constructor(seed) {
    this.s = BigInt(seed);
  }
  next() {
    this.s = (this.s + 0x9E3779B97F4A7C15n) & 0xFFFFFFFFFFFFFFFFn;
    let z = this.s;
    z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & 0xFFFFFFFFFFFFFFFFn;
    z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & 0xFFFFFFFFFFFFFFFFn;
    return (z ^ (z >> 31n)) & 0xFFFFFFFFFFFFFFFFn;
  }
  uni(a, b) {
    const r = Number(this.next() >> 11n) * (1.0 / 9007199254740992.0);
    return a + (b - a) * r;
  }
}

// Generate a synthetic grid matching powerflow.h
function generateSyntheticGrid(n = 100, seed = 2026) {
  const rng = new Rng(seed + n);
  const s = Math.ceil(Math.sqrt(n));
  const baseMVA = 100.0;

  const buses = [];
  for (let i = 0; i < n; i++) {
    const x = i % s;
    const y = Math.floor(i / s);
    const pd = rng.uni(2.0, 10.0);
    const qd = pd * rng.uni(0.2, 0.4);
    buses.push({
      id: i + 1,
      type: 1, // default PQ
      Pd: parseFloat(pd.toFixed(2)),
      Qd: parseFloat(qd.toFixed(2)),
      Gs: 0,
      Bs: 0,
      Vm0: 1.0,
      Va0: 0.0,
      x: 100 + x * 90 + rng.uni(-15, 15),
      y: 100 + y * 90 + rng.uni(-15, 15)
    });
  }

  const branches = [];
  const addLine = (a, b) => {
    const x = rng.uni(0.02, 0.06);
    const r = x / rng.uni(8.0, 15.0);
    const lineB = rng.uni(0.01, 0.03);
    branches.push({
      from: a + 1,
      to: b + 1,
      r: parseFloat(r.toFixed(5)),
      x: parseFloat(x.toFixed(5)),
      b: parseFloat(lineB.toFixed(5)),
      tap: 0,
      status: 1
    });
  };

  for (let i = 0; i < n; i++) {
    const x = i % s;
    if (x + 1 < s && i + 1 < n) addLine(i, i + 1);
    if (i + s < n && (x === 0 || rng.uni(0, 1) < 0.5)) addLine(i, i + s);
  }

  const extra = Math.floor(n / 20);
  for (let k = 0; k < extra; k++) {
    const i = Number(rng.next() % BigInt(n));
    const dx = Math.floor(rng.uni(-3, 4));
    const dy = Math.floor(rng.uni(-3, 4));
    const x = (i % s) + dx;
    const y = Math.floor(i / s) + dy;
    if (x < 0 || x >= s || y < 0) continue;
    const j = y * s + x;
    if (j >= n || j === i) continue;
    addLine(Math.min(i, j), Math.max(i, j));
  }

  // Slack bus in center
  const slack = Math.min(n - 1, Math.floor(s / 2) * s + Math.floor(s / 2));
  buses[slack].type = 3;
  buses[slack].Vm0 = 1.02;

  const src = [slack];
  const gens = [{ bus: slack + 1, Pg: 0, Vg: 1.02 }];

  for (let i = 0; i < n; i++) {
    if (i !== slack && ((i % s) + 2 * Math.floor(i / s)) % 3 === 0) {
      buses[i].type = 2; // PV generator
      const vset = parseFloat(rng.uni(1.015, 1.025).toFixed(4));
      buses[i].Vm0 = vset;
      src.push(i);
      gens.push({ bus: i + 1, Pg: 0, Vg: vset });
    }
  }

  // Regional dispatch via BFS
  const adj = Array.from({ length: n }, () => []);
  for (const br of branches) {
    adj[br.from - 1].push(br.to - 1);
    adj[br.to - 1].push(br.from - 1);
  }

  const owner = Array(n).fill(-1);
  const queue = [...src];
  for (const s0 of src) owner[s0] = s0;

  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    for (const nbr of adj[cur]) {
      if (owner[nbr] < 0) {
        owner[nbr] = owner[cur];
        queue.push(nbr);
      }
    }
  }

  // Assign load to regional generator
  for (let i = 0; i < n; i++) {
    const gIndex = owner[i];
    if (gIndex >= 0 && gIndex !== slack) {
      const gObj = gens.find(g => g.bus === gIndex + 1);
      if (gObj) {
        gObj.Pg += buses[i].Pd;
      }
    }
  }

  // Format Pg
  for (const g of gens) {
    g.Pg = parseFloat(g.Pg.toFixed(2));
  }

  return {
    name: `Synthetic ${n}-Bus Grid`,
    baseMVA,
    buses,
    gens,
    branches
  };
}

module.exports = {
  parseCaseFile,
  serializeCaseFile,
  getIEEE14Case,
  getIEEE30Case,
  generateSyntheticGrid
};
