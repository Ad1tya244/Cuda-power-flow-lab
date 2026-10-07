# CUDA Case Study - Smart Energy Grids (GPU-Accelerated Power-Flow Simulation Lab)

---

## Executive Summary

Modern power grid transmission operators must solve the nonlinear AC power-flow equations every few seconds to monitor grid security, detect contingency vulnerabilities, and prevent catastrophic cascading blackouts. As intermittent renewable energy resources and distributed generation proliferate, electrical networks become increasingly dynamic, requiring orders-of-magnitude faster simulation capabilities than traditional single-threaded CPU implementations can deliver.

This project delivers a high-performance, GPU-accelerated AC power-flow simulation engine and interactive web laboratory. It features:
* A native **CUDA C++ solver** utilizing **cuSPARSE**, **cuBLAS**, and custom kernels for sparse linear algebra and nonlinear iterations.
* **Multicolour graph vertex colouring** to eliminate data dependencies in parallel incomplete LU ($\text{ILU}(0)$) triangular solves.
* A decoupled **Cloud Compute Gateway** architecture enabling cross-platform access from any standard client machine.
* A professional, **viewport-locked interactive SaaS dashboard** providing live grid visualization, hardware execution tracing, and real-time Jacobian matrix sparsity telemetry.

---

## Mathematical Formulation

### AC Power-Flow Equations

Electrical transmission networks are modeled by non-linear algebraic power balance equations. For an $N$-bus network, the complex voltage at bus $i$ in polar coordinates is:

$$V_i = |V_i| e^{j \theta_i}$$

The nodal admittance matrix elements are defined as $Y_{ik} = G_{ik} + j B_{ik}$. The active ($P_i$) and reactive ($Q_i$) power injections at bus $i$ are:

$$P_i(V, \theta) = \sum_{k=1}^N |V_i| |V_k| \left( G_{ik} \cos(\theta_i - \theta_k) + B_{ik} \sin(\theta_i - \theta_k) \right)$$

$$Q_i(V, \theta) = \sum_{k=1}^N |V_i| |V_k| \left( G_{ik} \sin(\theta_i - \theta_k) - B_{ik} \cos(\theta_i - \theta_k) \right)$$

### Bus Classifications

1. **Slack / Swing Bus (Bus 1)**: Voltage magnitude $|V_1|$ and reference angle $\theta_1 = 0^\circ$ are fixed. Rebalances system losses.
2. **Generator / PV Buses**: Active power generation $P_{g,i}$ and voltage setpoint $|V_i|$ are specified. Unknowns: $\theta_i, Q_i$.
3. **Load / PQ Buses**: Active load $P_{d,i}$ and reactive load $Q_{d,i}$ are specified. Unknowns: $\theta_i, |V_i|$.

### Newton-Raphson Solution Method

The state vector of unknowns is:

$$x = \begin{bmatrix} \boldsymbol{\theta} \\ |\mathbf{V}| \end{bmatrix}$$

The power mismatches at iteration $k$ are defined as:

$$\Delta P_i = P_i^{\text{spec}} - P_i(x^{(k)})$$

$$\Delta Q_i = Q_i^{\text{spec}} - Q_i(x^{(k)})$$

$$F(x^{(k)}) = \begin{bmatrix} \boldsymbol{\Delta P} \\ \boldsymbol{\Delta Q} \end{bmatrix}$$

At each Newton-Raphson iteration, the linear system is solved for state corrections $\Delta x^{(k)}$:

$$J(x^{(k)}) \Delta x^{(k)} = -F(x^{(k)})$$

$$x^{(k+1)} = x^{(k)} + \Delta x^{(k)}$$

The Jacobian matrix $J$ has the 4-block structure:

$$J = \begin{bmatrix} 
\dfrac{\partial P}{\partial \theta} & \dfrac{\partial P}{\partial |V|} \\[10pt]
\dfrac{\partial Q}{\partial \theta} & \dfrac{\partial Q}{\partial |V|} 
\end{bmatrix}$$

Iteration terminates when the maximum absolute power mismatch satisfies:

$$\|F(x^{(k)})\|_\infty = \max_i \left( |\Delta P_i|, |\Delta Q_i| \right) < 10^{-8} \text{ p.u.}$$

---

## GPU-Accelerated 5-Stage CUDA Architecture

Electrical power grids exhibit extreme sparsity (>99% zero entries in $Y_{bus}$ because each bus connects to an average of only 2 to 4 adjacent lines). Storing and factoring dense matrices introduces quadratic memory overhead and cubic operation counts. This implementation executes the Newton-Raphson pipeline directly in Compressed Sparse Row (CSR) format across 5 execution stages (6 pipeline steps):

```
+-----------------------------------------------------------------------------------+
|                            CUDA 5-STAGE / 6-STEP PIPELINE                         |
+-----------------------------------------------------------------------------------+
|  Step 1: k_voltage       Convert polar state vectors (Vm, Va) -> rectangular (Vr,Vi)  |
|  Step 2: cusparseSpMV    Compute injected currents: I = Ybus * V (touch non-zeros)   |
|  Step 3: k_mismatch      Evaluate active & reactive power residuals: F = V * conj(I) - S |
|  Step 4: k_jacobian      Assemble CSR non-zero Jacobian derivatives directly on GPU |
|  Step 5: ILU + BiCGSTAB  Compute csrilu02 preconditioner + colour-scheduled solve   |
|  Step 6: k_update        Apply state vector correction: x^(k+1) = x^(k) + dx (axpy)  |
+-----------------------------------------------------------------------------------+
```

### Stage Breakdown

1. **Stage 1 (Voltage State Initialization)**:
   * Kernel: `k_voltage` (1 thread per bus).
   * Transforms polar voltage magnitudes and phase angles into rectangular coordinates on the device without host-device memory transfers.

2. **Stage 2 (Complex Current Injections)**:
   * Routine: `cusparseSpMV` (cuSPARSE Library).
   * Evaluates complex nodal current injections $I = Y_{bus} V$ using high-throughput sparse matrix-vector multiplication.

3. **Stage 3 (Power Residuals / Mismatch Evaluation)**:
   * Kernel: `k_mismatch` (1 thread per equation).
   * Evaluates active power residuals $\Delta P$ and reactive power residuals $\Delta Q$. Computes the $\ell_\infty$ norm on device to verify convergence.

4. **Stage 4 (Direct Sparse Jacobian Assembly)**:
   * Kernel: `k_jacobian` (1 thread per transmission branch).
   * Calculates off-diagonal and diagonal derivative entries directly into pre-allocated CSR column index and value buffers, completely bypassing dense matrix construction.

5. **Stage 5 (Linear Solution via ILU Preconditioned BiCGSTAB)**:
   * Preconditioning: `cusparseDcsrilu02` generates an incomplete lower/upper factorization $\text{ILU}(0)$ preserving the sparsity pattern.
   * Linear Solver: Biconjugate Gradient Stabilized (BiCGSTAB) iterates using `cublasDdot` and `cublasDaxpy`.
   * Parallel Triangular Solve: Uses level-scheduled multicolour graph coloring (`k_lsolve_colour`) to execute backward and forward substitutions in parallel across CUDA warps.

6. **Stage 6 (State Update)**:
   * Kernel: `k_update` (cuBLAS `axpy`).
   * Updates state variables $\theta$ and $|V|$ concurrently, checking voltage limit bounds and updating phase angles.

---

## Multicolour Graph Vertex Colouring

### The Challenge
Incomplete LU triangular substitution ($L y = b$ and $U x = y$) is traditionally sequential because solving for variable $i$ depends on previously solved adjacent variables $k < i$. This serial dependency creates data hazards when attempted in parallel on GPU architectures.

### The Solution
The electrical network graph is pre-processed using a greedy multicolour vertex colouring algorithm:
1. Every bus in the network is assigned an integer colour such that no two adjacent connected buses share the same colour:
   $$\text{color}(i) \neq \text{color}(k) \quad \forall (i, k) \in \mathcal{E}$$
2. Typical power transmission networks are planar or near-planar sparse graphs, requiring only 4 to 6 independent colours.
3. During forward and backward triangular substitution, all buses belonging to colour group $c$ are solved simultaneously in parallel by CUDA threads with zero race conditions.
4. A thread block synchronisation barrier (`__syncthreads()`) is placed between successive colour steps, guaranteeing mathematical correctness and complete execution independence.

---

## Client/Server Architecture

```
+-------------------------------------------------------------+
|                BROWSER FRONTEND (HTML5/CSS3/JS)             |
|  - Viewport-locked 100vh SaaS Dashboard (Zero Scrolling)     |
|  - Interactive IEEE 14-bus Canvas Topology                  |
|  - Real-Time Live 22x22 Jacobian Matrix Sparsity Heatmap    |
|  - Dynamic Load Slider, Operational Scenarios, Telemetry    |
+-------------------------------------------------------------+
                              |
                     WebSocket / REST API
                              |
+-------------------------------------------------------------+
|            NODE.JS COMPUTE GATEWAY (Port 3000)              |
|  - cases.js: IEEE 14/30 and synthetic benchmark models       |
|  - solver.js: Reference power-flow and state estimator      |
|  - contingency.js: Automated N-1 security screening engine  |
|  - cuda_bridge.js: Hardware interface & telemetry streamer  |
+-------------------------------------------------------------+
                              |
                   Native Execution / Tunnel
                              |
+-------------------------------------------------------------+
|               CUDA SOLVER / REMOTE GPU WORKER               |
|  - Local Native GPU: ./pf_cuda (NVIDIA Pascal/Turing/Ada)   |
|  - Remote Cloud GPU: Google Colab (Tesla T4 via Cloudflare) |
+-------------------------------------------------------------+
```

### Key Architectural Strengths

* **Presentation/Compute Decoupling**: The web browser strictly handles UI layout, interactive canvas rendering, and telemetry displays. Heavy matrix arithmetic and CUDA instructions are executed on the compute server or remote cloud GPU.
* **Universal Hardware Compatibility**: The lab runs smoothly on standard client hardware (including Apple Silicon macOS, Linux, and Windows) by delegating GPU workloads to the compute gateway or Google Colab.
* **Dual Operation Modes**: When a remote NVIDIA GPU is connected, calculations execute on cloud hardware. If offline, the gateway runs in high-fidelity local emulation mode with identical numerical precision and telemetry.

---

## Single-Command Launch

Launch the complete Cloud Compute Gateway, Simulation Engine, and Web Dashboard with a single command:

```bash
./run.sh
```

Alternative start commands:
```bash
npm start
# or:
make lab
```

### Launch Sequence
1. Initializes the Node.js API and WebSocket gateway on `http://localhost:3000` (auto-detecting alternative ports if 3000 is occupied).
2. Automatically opens your default web browser on macOS (`open`), Linux (`xdg-open`), or Windows (`start`).
3. Establishes the real-time simulation WebSocket connection with zero manual configuration.

### Command Flags
* Launch without opening a browser window: `./run.sh --no-open`
* Launch on a custom port: `./run.sh 8080`
* Run verification test suite: `npm test`

---

## Interactive GridLab Web Dashboard

The web interface is engineered as a high-density, professional engineering SaaS control center:

1. **Consolidated Top Toolbar**:
   * **Scenario Selector**: Instant switching between operational contingencies.
   * **Interactive Load Slider**: Dynamic active/reactive demand scaling from $0.5\times$ to $3.0\times$ with live physics recalculation.
   * **Simulation Controls**: One-click Run Simulation, single-step CUDA kernel execution (Step), and grid state Reset.
   * **Cloud GPU Status Cluster**: Displays connection state (Cloud GPU Online), device badge (NVIDIA T4), allocated VRAM (16 GB), and Configuration settings modal aligned to the far right.

2. **IEEE 14-Bus Interactive Network Topology**:
   * High-DPI canvas supporting drag-and-drop bus repositioning, smooth zoom, and pan.
   * Live bus markers colored by role: Slack (Blue), Generator/PV (Green), Load/PQ (Orange).
   * Real-time line power flow labels (MW throughput) and bus voltage readings (p.u.).
   * Voltage boundary rings highlighting nominal, warning, and critical violation thresholds.

3. **6-Step CUDA Execution Pipeline Strip**:
   * Horizontal visual pipeline tracking kernel progress: `1 Voltage` (`k_voltage`), `2 SpMV` (`cusparseSpMV`), `3 Mismatch` (`k_mismatch`), `4 Jacobian` (`k_jacobian`), `5 Solve` (`csrilu02` + BiCGSTAB), and `6 Update` (`cublas axpy`).
   * Real-time warp thread indicators and stage description logs.

4. **Simulation Status Telemetry**:
   * Converged / Diverged status badge.
   * Newton-Raphson iteration count (with visual target bar).
   * Elapsed solve time (ms).
   * Maximum mismatch residual ($\|F\|_\infty$).

5. **System Summary KPIs**:
   * Total Active Generation (MW).
   * Total Grid Demand (MW).
   * System Active Transmission Losses (MW).
   * Active Voltage Violations count.

6. **Bus Voltages Bar Chart**:
   * Real-time bar chart showing per-bus voltage profile across all 14 buses.
   * Reference indicators for nominal operating band ($0.95$ to $1.05$ p.u.).
   * Dynamic color shifting for under-voltage (< 0.95 p.u.) and over-voltage (> 1.05 p.u.).

7. **Live 22x22 Jacobian Matrix Sparsity Heatmap**:
   * Razor-sharp, high-DPI canvas rendering of the $22 \times 22$ Jacobian matrix ($146$ non-zero elements, $30.2\%$ fill factor).
   * Displays CSR sparsity structure with quadrant demarcation separating angle ($\theta$) and voltage magnitude ($V$) blocks.
   * Highlighted diagonal pivots reflecting incomplete LU preconditioning activity.
   * Perfectly aligned to the bottom boundary of the Grid Topology box for a clean, unified dashboard layout.

---

## Operational Scenarios

The simulator provides pre-configured operational scenarios for evaluating grid resilience:

1. **Baseline (Normal)**: Standard IEEE 14-bus base case with balanced generation and loading. All voltages remain in nominal range ($1.010$ to $1.090$ p.u.), with total active transmission losses of $13.8$ MW.
2. **Peak Demand (+30% Load)**: Scales active and reactive power demand across all 14 buses by $1.3\times$. Demonstrates increased transmission losses, reactive power depletion, and voltage sag across weak load buses (e.g., Bus 14).
3. **Trip PV Generator (Bus 2)**: Simulates the sudden loss of the primary photovoltaic generator at Bus 2. The slack generator at Bus 1 compensates for the $40.0$ MW generation shortfall.
4. **Critical Line Trip (L1-2)**: Simulates an $N-1$ contingency trip of transmission line 1-2 (the highest loaded corridor). Forces power re-routing through line 1-5, causing elevated line loading and increased reactive losses.
5. **Voltage Collapse Stress Test**: Heavily stresses load buses to drive the Jacobian matrix towards singularity, demonstrating solvability boundaries, high iteration counts, and ill-conditioned linear solves.

---

## Automated N-1 Contingency Analysis

The engine incorporates automated $N-1$ security screening across all transmission branches:

* Iteratively trips each transmission line in the network.
* Resolves the power-flow equations to determine post-contingency voltage and line loading conditions.
* Computes the composite **Performance Index ($PI$)** for each contingency:

$$PI = \sum_{i=1}^{N_{\text{bus}}} \left( \frac{|V_i - 1.0|}{0.05} \right)^2 + \sum_{k=1}^{N_{\text{branch}}} \left( \frac{\text{Flow}_k}{\text{Limit}_k} \right)^4$$

* Identifies islanding risks and ranks contingencies in descending order of severity.

---

## Remote Google Colab GPU Setup

To execute the native CUDA solver on a free remote NVIDIA Tesla T4 GPU in Google Colab:

1. Open [Google Colab](https://colab.research.google.com) and upload the notebook: [colab_gpu_worker.ipynb](colab_gpu_worker.ipynb).
2. Configure GPU accelerator: **Runtime -> Change runtime type -> T4 GPU**.
3. Execute the setup script in a notebook cell:
   ```bash
   !python3 tools/colab_setup.py
   ```
4. The automated setup will:
   * Confirm NVIDIA Tesla T4 GPU hardware presence via `nvidia-smi`.
   * Compile the CUDA solver (`make ARCH=sm_75`).
   * Launch the GPU worker service on port 8000.
   * Establish a Cloudflare tunnel and output your unique HTTPS endpoint:
     ```text
     ======================================================================
     REMOTE COLAB GPU COMPUTE WORKER IS LIVE
     ENDPOINT URL: https://your-tunnel-subdomain.trycloudflare.com
     ======================================================================
     ```
5. In the Web Simulation Dashboard, click the **Settings Gear** icon in the top toolbar, select `NVIDIA Tesla T4 (Google Colab)`, paste your public tunnel URL into the endpoint field, and click **Connect**.

---

## Native CUDA CLI Solver

For running directly on a Linux workstation or WSL environment equipped with an NVIDIA GPU:

### Compilation

```bash
# Compile binary for target architecture
make ARCH=sm_75              # sm_61 = Pascal, sm_75 = Tesla T4, sm_86 = RTX 30xx, sm_89 = RTX 40xx
```

### CLI Command Options

```bash
# Run IEEE 14-bus test with narrated 5-stage CUDA pipeline execution
./pf_cuda demo

# Run numerical correctness verification across synthetic grids (1k, 10k, 100k buses)
./pf_cuda verify

# Execute empirical scaling benchmarks (Experiments E1 through E5)
./pf_cuda bench

# Solve a custom IEEE or synthetic power-flow case file
./pf_cuda solve cases/ieee14.case
```

---

## Empirical Benchmarks & Scaling Results

Empirical timing measurements gathered across synthetic grid cases from 1,000 to 200,000 buses ([results/e1_final.csv](results/e1_final.csv) and [results/e2_phases.csv](results/e2_phases.csv)):

| Grid Size (Buses) | Branches | CPU Time (ms) | GPU Time (ms) | GPU Speedup |
|---|---|---|---|---|
| 1,000 | 1,430 | 1.82 | 4.12 | 0.44x |
| 5,000 | 7,150 | 11.45 | 8.30 | 1.38x |
| 10,000 | 14,300 | 28.60 | 14.10 | 2.03x |
| 50,000 | 71,500 | 184.20 | 56.40 | 3.27x |
| 100,000 | 143,000 | 412.50 | 112.30 | 3.67x |
| 200,000 | 286,000 | 985.10 | 234.80 | 4.19x |

### Observations
* **Break-Even Point**: Below ~5,000 buses, CPU execution is faster due to kernel launch latency and memory transfer overhead.
* **Large-Scale Regimes**: For networks with 50,000+ buses, the parallel CSR Jacobian assembly and multicolour triangular solves deliver greater than $3.5\times$ to $4.2\times$ wall-clock speedups over optimized single-threaded CPU implementations.

---

## Verification Test Suite

The repository includes an automated 31-step verification suite testing all core components:

```bash
npm test
# or: node tools/test_simulation.js
```

### Verification Output

```text
================================================================
  STARTING POWER FLOW SIMULATION LAB VERIFICATION SUITE
================================================================

Test 1: IEEE 14 Case & Sparse CSR Ybus Matrix
  [PASS] IEEE 14 has exactly 14 buses
  [PASS] IEEE 14 has exactly 20 branches
  [PASS] IEEE 14 has 5 generators
  [PASS] Ybus dimension is 14x14
  [PASS] Ybus nnz = 54 (expected 54 non-zeros)
  [PASS] CSR rowptr length is n + 1
  [PASS] CSR column index length matches nnz

Test 2: Multicolour Graph Coloring & Parallel Triangular Solves
  [PASS] Graph colored with 4 colors (GPU friendly <= 6)
  [PASS] Adjacent buses never share a color (zero race conditions on GPU)

Test 3: Newton-Raphson Power Flow & Accuracy Verification
  [PASS] Newton-Raphson converges successfully
  [PASS] Converged in 3 iterations (expected <= 4)
  [PASS] Final max mismatch = 3.427e-9 pu (< 1e-8 pu)
  [PASS] Max |dVm| vs published = 0.0021 pu (< 0.003 pu)
  [PASS] Max |dVa| vs published = 0.150 deg (< 0.2 deg)
  [PASS] Computed power flows for all 20 transmission lines
  [PASS] Total grid active power loss = 13.806 MW (> 0 MW)

Test 4: Remote Cloud GPU 5-Stage Pipeline Telemetry
  [PASS] CUDA pipeline execution returned successful result
  [PASS] Emitted STAGE_1_HOST_PREP
  [PASS] Emitted STAGE_1_H2D_MEMCPY
  [PASS] Emitted STAGE_3_NEWTON_SOLVER
  [PASS] Emitted STAGE_4_D2H_MEMCPY
  [PASS] Emitted STAGE_5_COMPLETE
  [PASS] Generated kernel execution logs for all Newton iterations

Test 5: Automated N-1 Contingency Screening
  [PASS] Evaluated all 20 single-line contingencies (N-1)
  [PASS] Ranked table has 20 contingency entries
  [PASS] Contingencies are ranked in descending order of severity (PI index)

Test 6: Scalability on Synthetic 100-Bus Grid
  [PASS] Synthetic grid generated with 100 buses
  [PASS] Generated 143 branches (meshed degree ~3)
  [PASS] Synthetic 100-bus solved in 3 iterations

Test 7: Benchmark Data Verification
  [PASS] Loaded empirical scaling data from results/e1_final.csv
  [PASS] Loaded phase breakdown data from results/e2_phases.csv

================================================================
  TEST RESULTS: 31 / 31 TESTS PASSED (100%)
================================================================
```

---

## Repository Structure

| Path | Purpose |
|---|---|
| `run.sh` | Single-command launcher (starts server and opens web browser) |
| `package.json` | Project configuration and test runner (`npm start`, `npm test`) |
| `.gitignore` | Ignore rules for OS files, dependencies, build binaries, and logs |
| `Makefile` | C++ and CUDA compilation targets (`make ARCH=sm_75`, `make lab`) |
| `colab_gpu_worker.ipynb` | Google Colab interactive notebook for remote GPU worker execution |
| `src/pf_cuda.cu` | Native CUDA kernels, cuSPARSE SpMV, and multicolour triangular solver |
| `src/powerflow.h` | Shared mathematical definitions, CSR matrices, and CPU baseline solver |
| `web/server/index.js` | HTTP server and RFC 6455 WebSocket gateway |
| `web/server/solver.js` | Newton-Raphson AC power-flow solver engine |
| `web/server/contingency.js` | Automated N-1 contingency analysis and ranking engine |
| `web/server/cuda_bridge.js` | Remote cloud GPU communication bridge and hardware telemetry |
| `web/server/cases.js` | IEEE 14, IEEE 30, and synthetic grid models parser and serializer |
| `web/public/index.html` | High-density SaaS control dashboard structure |
| `web/public/css/style.css` | Viewport-locked styles, typography, and responsive grid layout |
| `web/public/js/app.js` | Master client coordinator, WebSocket handler, and scenario manager |
| `web/public/js/grid_canvas.js` | Interactive IEEE 14-bus canvas renderer with auto-fit resizing |
| `web/public/js/spy_plot.js` | Admittance and Jacobian matrix CSR sparsity spy plot renderer |
| `web/public/js/pipeline_visualizer.js` | 5-stage CUDA data-path execution timeline |
| `web/public/js/contingency_view.js` | Contingency screening table and outage controller |
| `web/public/js/scaling_chart.js` | CPU vs. GPU empirical performance and scaling charts |
| `cases/ieee14.case` | Standard IEEE 14-bus test network specification |
| `tools/cloud_gpu_worker.py` | Standalone Python REST worker for remote GPU execution |
| `tools/colab_setup.py` | Automated Colab environment configuration script with Cloudflare tunnel |
| `tools/test_simulation.js` | 31-step verification suite for power-flow mathematics and telemetry |
| `tools/pf_reference.py` | Independent SciPy SuperLU reference verification solver |
| `tools/diagram.py` | Network graph layout generation utility |
| `tools/plot.py` | Benchmark scaling chart visualization generator |
| `results/e1_final.csv` | Empirical timing measurements from 1k to 200k bus grids |
| `results/e2_phases.csv` | Kernel phase timing breakdown data |

---

## References

1. J. J. Grainger and W. D. Stevenson, *Power System Analysis*, McGraw-Hill, 1994.
2. NVIDIA Corporation, *cuSPARSE Library User Guide*, 2023.
3. NVIDIA Corporation, *cuBLAS Library User Guide*, 2023.
4. Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
5. IEEE PES Power Systems Test Case Archive, *14 Bus Power Flow Test Case*, University of Washington.
