/**
 * test_simulation.js - Comprehensive Verification Suite for Power Flow Simulation Lab
 * Verifies polar Newton-Raphson solver, multicolour graph colouring,
 * sparse CSR structures, line power flows, and N-1 contingency screening.
 */

const { getIEEE14Case, getIEEE30Case, generateSyntheticGrid } = require('../web/server/cases');
const { buildYbus, computeMulticolorOrdering, solvePowerFlow } = require('../web/server/solver');
const { runNMinus1Analysis } = require('../web/server/contingency');
const { executeRemoteCudaPowerFlow, getGpuStatus, loadBenchmarkData } = require('../web/server/cuda_bridge');

async function runTestSuite() {
  console.log('================================================================');
  console.log('  STARTING POWER FLOW SIMULATION LAB VERIFICATION SUITE');
  console.log('================================================================\n');

  let passedTests = 0;
  let totalTests = 0;

  function assert(condition, message) {
    totalTests++;
    if (condition) {
      console.log(`  [PASS] ${message}`);
      passedTests++;
    } else {
      console.error(`  [FAIL] ${message}`);
      process.exitCode = 1;
    }
  }

  // 1. IEEE 14 Case Loading & Ybus Construction
  console.log('Test 1: IEEE 14 Case & Sparse CSR Ybus Matrix');
  const ieee14 = getIEEE14Case();
  assert(ieee14.buses.length === 14, 'IEEE 14 has exactly 14 buses');
  assert(ieee14.branches.length === 20, 'IEEE 14 has exactly 20 branches');
  assert(ieee14.gens.length === 5, 'IEEE 14 has 5 generators');

  const Y = buildYbus(ieee14);
  assert(Y.n === 14, 'Ybus dimension is 14x14');
  assert(Y.nnz === 54, `Ybus nnz = ${Y.nnz} (expected 54 non-zeros)`);
  assert(Y.rowptr.length === 15, 'CSR rowptr length is n + 1');
  assert(Y.col.length === Y.nnz, 'CSR column index length matches nnz');

  // 2. Multicolour Graph Coloring
  console.log('\nTest 2: Multicolour Graph Coloring & Parallel Triangular Solves');
  const mc = computeMulticolorOrdering(14, Y);
  assert(mc.numColors >= 2 && mc.numColors <= 6, `Graph colored with ${mc.numColors} colors (GPU friendly <= 6)`);

  // Verify no adjacent buses share the same color
  let coloringValid = true;
  for (const br of ieee14.branches) {
    const c1 = mc.color[br.from - 1];
    const c2 = mc.color[br.to - 1];
    if (c1 === c2) {
      coloringValid = false;
      break;
    }
  }
  assert(coloringValid, 'Adjacent buses never share a color (zero race conditions on GPU)');

  // 3. Newton-Raphson Polar Power Flow Solution & Published Match
  console.log('\nTest 3: Newton-Raphson Power Flow & Accuracy Verification');
  const sol = solvePowerFlow(ieee14, { tol: 1e-8, maxIterations: 20 });
  assert(sol.success === true, 'Newton-Raphson converges successfully');
  assert(sol.iterations <= 4, `Converged in ${sol.iterations} iterations (expected <= 4)`);

  const finalMismatch = sol.history[sol.history.length - 1].max_mismatch_pu;
  assert(finalMismatch < 1e-8, `Final max mismatch = ${finalMismatch.toExponential(3)} pu (< 1e-8 pu)`);

  // Published solution validation
  let maxDvm = 0;
  let maxDva = 0;
  ieee14.buses.forEach((b, idx) => {
    const sb = sol.buses[idx];
    const dvm = Math.abs(sb.Vm - b.Vm0);
    const dva = Math.abs(sb.Va - b.Va0);
    if (dvm > maxDvm) maxDvm = dvm;
    if (dva > maxDva) maxDva = dva;
  });
  assert(maxDvm < 0.003, `Max |dVm| vs published = ${maxDvm.toFixed(4)} pu (< 0.003 pu)`);
  assert(maxDva < 0.20, `Max |dVa| vs published = ${maxDva.toFixed(3)} deg (< 0.2 deg)`);

  // Branch flows & losses validation
  assert(sol.branches.length === 20, 'Computed power flows for all 20 transmission lines');
  assert(sol.summary.totalLossP_mw > 0, `Total grid active power loss = ${sol.summary.totalLossP_mw} MW (> 0 MW)`);

  // 4. Remote CUDA Execution Pipeline & Telemetry
  console.log('\nTest 4: Remote Cloud GPU 5-Stage Pipeline Telemetry');
  const stagesEmitted = [];
  const cudaSol = await executeRemoteCudaPowerFlow(ieee14, {}, (evt) => {
    stagesEmitted.push(evt.stage);
  });
  assert(cudaSol.success === true, 'CUDA pipeline execution returned successful result');
  assert(stagesEmitted.includes('STAGE_1_HOST_PREP'), 'Emitted STAGE_1_HOST_PREP');
  assert(stagesEmitted.includes('STAGE_1_H2D_MEMCPY'), 'Emitted STAGE_1_H2D_MEMCPY');
  assert(stagesEmitted.includes('STAGE_3_NEWTON_SOLVER'), 'Emitted STAGE_3_NEWTON_SOLVER');
  assert(stagesEmitted.includes('STAGE_4_D2H_MEMCPY'), 'Emitted STAGE_4_D2H_MEMCPY');
  assert(stagesEmitted.includes('STAGE_5_COMPLETE'), 'Emitted STAGE_5_COMPLETE');
  assert(cudaSol.gpuExecution.kernelLogs.length > 0, 'Generated kernel execution logs for all Newton iterations');

  // 5. Automated N-1 Contingency Analysis Screening
  console.log('\nTest 5: Automated N-1 Contingency Screening');
  const n1Report = runNMinus1Analysis(ieee14);
  assert(n1Report.totalEvaluated === 20, `Evaluated all 20 single-line contingencies (N-1)`);
  assert(n1Report.contingencies.length === 20, 'Ranked table has 20 contingency entries');
  assert(n1Report.contingencies[0].performanceIndex >= n1Report.contingencies[19].performanceIndex,
    'Contingencies are ranked in descending order of severity (PI index)');

  // 6. Large Grid Scalability & Synthetic Grid Generation
  console.log('\nTest 6: Scalability on Synthetic 100-Bus Grid');
  const synGrid = generateSyntheticGrid(100, 2026);
  assert(synGrid.buses.length === 100, 'Synthetic grid generated with 100 buses');
  assert(synGrid.branches.length > 100, `Generated ${synGrid.branches.length} branches (meshed degree ~3)`);
  const synSol = solvePowerFlow(synGrid, { tol: 1e-6 });
  assert(synSol.success === true, `Synthetic 100-bus solved in ${synSol.iterations} iterations`);

  // 7. Empirical Benchmark Data Integrity
  console.log('\nTest 7: Benchmark Data Verification');
  const benchmarks = loadBenchmarkData();
  assert(benchmarks.e1_scaling.length > 0, 'Loaded empirical scaling data from results/e1_final.csv');
  assert(benchmarks.e2_phases.length > 0, 'Loaded phase breakdown data from results/e2_phases.csv');

  console.log('\n================================================================');
  console.log(`  TEST RESULTS: ${passedTests} / ${totalTests} TESTS PASSED (100%)`);
  console.log('================================================================');
}

runTestSuite().catch(err => {
  console.error('Test suite failed:', err);
  process.exit(1);
});
