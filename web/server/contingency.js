/**
 * contingency.js - Automated N-1 Contingency Analysis Engine
 * Screens single-line outages (N-1), computes post-contingency power flow,
 * evaluates voltage security limits, and ranks critical contingencies.
 */

const { solvePowerFlow } = require('./solver');

/**
 * Checks if grid is connected via BFS
 */
function isConnected(grid, outagedBranchIdx) {
  const n = grid.buses.length;
  const adj = Array.from({ length: n }, () => []);
  const idMap = new Map();
  grid.buses.forEach((b, idx) => idMap.set(b.id, idx));

  for (let i = 0; i < grid.branches.length; i++) {
    if (i === outagedBranchIdx || grid.branches[i].status === 0) continue;
    const f = idMap.get(grid.branches[i].from);
    const t = idMap.get(grid.branches[i].to);
    if (f !== undefined && t !== undefined) {
      adj[f].push(t);
      adj[t].push(f);
    }
  }

  const visited = new Set();
  const queue = [0];
  visited.add(0);

  while (queue.length > 0) {
    const cur = queue.shift();
    for (const nbr of adj[cur]) {
      if (!visited.has(nbr)) {
        visited.add(nbr);
        queue.push(nbr);
      }
    }
  }

  return visited.size === n;
}

/**
 * Runs N-1 Contingency Screening on all transmission lines
 */
function runNMinus1Analysis(grid, options = {}) {
  const tStart = Date.now();
  const contingencies = [];
  const totalBranches = grid.branches.length;

  for (let bIdx = 0; bIdx < totalBranches; bIdx++) {
    const targetBranch = grid.branches[bIdx];
    if (targetBranch.status === 0) continue; // Skip lines already outaged

    const branchName = `L${targetBranch.from}-${targetBranch.to}`;
    const connected = isConnected(grid, bIdx);

    if (!connected) {
      contingencies.push({
        branchIndex: bIdx,
        branchName,
        from: targetBranch.from,
        to: targetBranch.to,
        status: 'ISLANDING_RISK',
        severity: 'CRITICAL',
        converged: false,
        performanceIndex: 999999,
        maxVoltageDeviation: 0.5,
        worstBus: null,
        underVoltageCount: 0,
        overVoltageCount: 0,
        maxLoadingPct: 0,
        overloadedBranches: 0,
        message: 'Tripping this branch isolates grid buses (Islanding/Blackout).'
      });
      continue;
    }

    // Clone grid with target branch outaged
    const candGrid = {
      ...grid,
      branches: grid.branches.map((br, idx) => ({
        ...br,
        status: idx === bIdx ? 0 : br.status
      }))
    };

    // Remotely solve post-contingency power flow
    const sol = solvePowerFlow(candGrid, { maxIterations: 15, tol: 1e-6 });

    if (!sol.success) {
      contingencies.push({
        branchIndex: bIdx,
        branchName,
        from: targetBranch.from,
        to: targetBranch.to,
        status: 'VOLTAGE_COLLAPSE',
        severity: 'CRITICAL',
        converged: false,
        performanceIndex: 888888,
        maxVoltageDeviation: 0.5,
        worstBus: null,
        underVoltageCount: 0,
        overVoltageCount: 0,
        maxLoadingPct: 0,
        overloadedBranches: 0,
        message: 'Newton-Raphson diverges: post-contingency voltage collapse.'
      });
      continue;
    }

    // Inspect bus voltages
    let maxVDev = 0;
    let worstBusId = null;
    let underCount = 0;
    let overCount = 0;
    let piVoltage = 0;

    for (const b of sol.buses) {
      const dev = Math.abs(b.Vm - 1.0);
      if (dev > maxVDev) {
        maxVDev = dev;
        worstBusId = b.id;
      }
      if (b.Vm < 0.95) underCount++;
      if (b.Vm > 1.05) overCount++;
      // Quadratic penalty outside 0.95 - 1.05
      if (b.Vm < 0.95 || b.Vm > 1.05) {
        const excess = Math.max(0.95 - b.Vm, b.Vm - 1.05);
        piVoltage += Math.pow(excess / 0.05, 2) * 50;
      }
    }

    // Inspect line loadings
    let maxLoadPct = 0;
    let overloadedCount = 0;
    let piOverload = 0;

    for (const br of sol.branches) {
      if (br.status === 'outage') continue;
      if (br.loading_pct > maxLoadPct) {
        maxLoadPct = br.loading_pct;
      }
      if (br.loading_pct > 100.0) {
        overloadedCount++;
        piOverload += Math.pow(br.loading_pct / 100.0, 4) * 20;
      }
    }

    const performanceIndex = parseFloat((piVoltage + piOverload).toFixed(2));

    let severity = 'SECURE';
    let statusText = 'Normal post-contingency operation';
    if (underCount > 0 || overCount > 0 || overloadedCount > 0) {
      if (maxVDev > 0.10 || maxLoadPct > 120.0 || underCount >= 2) {
        severity = 'CRITICAL';
        statusText = 'Severe voltage/thermal violations';
      } else {
        severity = 'WARNING';
        statusText = 'Moderate voltage/thermal warning';
      }
    }

    contingencies.push({
      branchIndex: bIdx,
      branchName,
      from: targetBranch.from,
      to: targetBranch.to,
      status: statusText,
      severity,
      converged: true,
      performanceIndex,
      maxVoltageDeviation: parseFloat(maxVDev.toFixed(4)),
      worstBus: worstBusId,
      underVoltageCount: underCount,
      overVoltageCount: overCount,
      maxLoadingPct: maxLoadPct,
      overloadedBranches: overloadedCount,
      solveIters: sol.iterations,
      message: `${underCount} under-voltage, ${overloadedCount} overloaded lines.`
    });
  }

  // Sort by severity (Performance Index descending)
  contingencies.sort((a, b) => b.performanceIndex - a.performanceIndex);

  const durationMs = Date.now() - tStart;

  return {
    totalEvaluated: contingencies.length,
    criticalCount: contingencies.filter(c => c.severity === 'CRITICAL').length,
    warningCount: contingencies.filter(c => c.severity === 'WARNING').length,
    secureCount: contingencies.filter(c => c.severity === 'SECURE').length,
    durationMs,
    contingencies
  };
}

module.exports = {
  runNMinus1Analysis
};
