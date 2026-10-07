/**
 * pipeline_visualizer.js - 5-Stage GPU CUDA Data-Path & Kernel Execution Inspector
 * Visualizes cuSPARSE, cuBLAS, custom CUDA kernels, and phase execution timings.
 */

class PipelineVisualizer {
  constructor(containerElement, consoleElement) {
    this.container = containerElement;
    this.consoleBox = consoleElement;
    this.currentStages = [];
    this.initDefaultStages();
  }

  initDefaultStages() {
    this.stages = [
      {
        id: 1,
        title: 'PCIe Host-to-Device Memory Setup',
        subtitle: 'cudaMalloc & cudaMemcpy H->D',
        desc: 'Allocates GPU VRAM for CSR Ybus, Jacobian patterns, and vector states. Streams initial voltages over PCIe.',
        calls: ['cudaMalloc(d_Y_rowptr)', 'cudaMalloc(d_V)', 'cudaMemcpyAsync(H2D)'],
        status: 'ready'
      },
      {
        id: 2,
        title: 'Multicolour Graph Colouring & Level Scheduling',
        subtitle: 'Greedy Bus Colouring (4-8 colours)',
        desc: 'Colours bus graph so adjacent buses never share a colour. Enables parallel triangular solves on GPU.',
        calls: ['build_problem()', 'cstart[]', 'brow0[]', 'bcnt[]'],
        status: 'ready'
      },
      {
        id: 3,
        title: 'GPU Newton-Raphson Nonlinear Loop',
        subtitle: 'cuSPARSE SpMV + Custom Kernels + BiCGSTAB',
        desc: 'Executes parallel polar mismatch evaluation, Jacobian values, cuSPARSE csrilu02, and multicolour BiCGSTAB.',
        calls: ['k_voltage', 'cusparseSpMV', 'k_mismatch', 'k_jacobian', 'cusparseDcsrilu02', 'k_lsolve_colour', 'cublasDdot', 'k_update'],
        status: 'ready'
      },
      {
        id: 4,
        title: 'PCIe Device-to-Host Voltage Transfer',
        subtitle: 'cudaMemcpy D->H',
        desc: 'Transfers converged voltage magnitudes Vm and phase angles Va from GPU VRAM back to host memory.',
        calls: ['cudaMemcpy(d_Vm, h_Vm, D2H)', 'cudaMemcpy(d_Va, h_Va, D2H)'],
        status: 'ready'
      },
      {
        id: 5,
        title: 'Memory Cleanup & Browser API Dispatch',
        subtitle: 'cudaFree & WebSocket Telemetry Stream',
        desc: 'Releases GPU device buffers and streams load-flow solution telemetry to the browser dashboard.',
        calls: ['cudaFree()', 'ws.send(SOLVE_COMPLETE)'],
        status: 'ready'
      }
    ];
    this.render();
  }

  setStageActive(stageIndex) {
    this.stages.forEach((st, idx) => {
      if (idx < stageIndex) st.status = 'completed';
      else if (idx === stageIndex) st.status = 'active';
      else st.status = 'ready';
    });
    this.render();
  }

  setAllCompleted() {
    this.stages.forEach(st => st.status = 'completed');
    this.render();
  }

  logConsole(message, type = 'info', kernel = null) {
    if (!this.consoleBox) return;
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0] + '.' + String(now.getMilliseconds()).padStart(3, '0');

    const entry = document.createElement('div');
    entry.className = 'console-entry';

    let kernelTag = kernel ? `<span class="kernel">[${kernel}]</span> ` : '';
    let colorStyle = type === 'error' ? 'color: #ff1744;' : (type === 'stage' ? 'color: #00e5ff;' : '');

    entry.innerHTML = `<span class="time">${timeStr}</span> ${kernelTag}<span style="${colorStyle}">${message}</span>`;
    this.consoleBox.appendChild(entry);
    this.consoleBox.scrollTop = this.consoleBox.scrollHeight;
  }

  clearConsole() {
    if (this.consoleBox) this.consoleBox.innerHTML = '';
  }

  render() {
    if (!this.container) return;
    this.container.innerHTML = '';

    this.stages.forEach(st => {
      const card = document.createElement('div');
      card.className = `stage-step-card ${st.status}`;

      const tagsHtml = st.calls.map(c => `<span class="cuda-call-tag">${c}</span>`).join('');

      card.innerHTML = `
        <div class="stage-header">
          <div class="stage-title">
            <div class="stage-number">${st.id}</div>
            ${st.title}
          </div>
          <span class="stage-badge">${st.status.toUpperCase()}</span>
        </div>
        <div class="stage-desc">${st.desc}</div>
        <div class="stage-cuda-tags">${tagsHtml}</div>
      `;

      this.container.appendChild(card);
    });
  }

  renderPhaseChart(canvasElement, phaseBreakdowns) {
    if (phaseBreakdowns) {
      this.lastPhaseBreakdowns = phaseBreakdowns;
    }
    const breakdowns = phaseBreakdowns || this.lastPhaseBreakdowns;
    if (!canvasElement || !breakdowns || breakdowns.length === 0) return;
    const parent = canvasElement.parentElement;
    if (!parent || parent.clientWidth <= 0) return;
    const ctx = canvasElement.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(100, parent.clientWidth - 24);
    const h = 170;
    canvasElement.width = w * dpr;
    canvasElement.height = h * dpr;
    canvasElement.style.width = `${w}px`;
    canvasElement.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, w, h);

    // Sum phase times across Newton iterations
    let totals = { inject: 0, jacobian: 0, ilu: 0, solve: 0, update: 0 };
    breakdowns.forEach(p => {
      totals.inject += p.inject_ms || 0;
      totals.jacobian += p.jacobian_ms || 0;
      totals.ilu += p.ilu_ms || 0;
      totals.solve += p.solve_ms || 0;
      totals.update += p.update_ms || 0;
    });

    const phases = [
      { name: 'SpMV & Injections', val: totals.inject, color: '#00e5ff' },
      { name: 'k_jacobian', val: totals.jacobian, color: '#00e676' },
      { name: 'csrilu02 ILU(0)', val: totals.ilu, color: '#ffab00' },
      { name: 'BiCGSTAB Solves', val: totals.solve, color: '#b388ff' },
      { name: 'k_update', val: totals.update, color: '#2979ff' }
    ];

    const maxVal = Math.max(...phases.map(p => p.val), 0.01);
    const barWidth = (w - 60) / phases.length;

    ctx.strokeStyle = 'rgba(70, 100, 150, 0.2)';
    ctx.beginPath();
    ctx.moveTo(35, h - 30);
    ctx.lineTo(w - 10, h - 30);
    ctx.stroke();

    phases.forEach((p, idx) => {
      const bx = 40 + idx * barWidth;
      const barH = (p.val / maxVal) * (h - 65);
      const by = h - 30 - barH;

      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 6;
      ctx.fillRect(bx + 4, by, barWidth - 12, barH);
      ctx.shadowBlur = 0;

      // Value label
      ctx.fillStyle = '#f1f5f9';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(`${p.val.toFixed(2)}ms`, bx + barWidth / 2, by - 6);

      // Category label
      ctx.fillStyle = '#94a3b8';
      ctx.font = '9.5px Inter';
      ctx.fillText(p.name, bx + barWidth / 2, h - 14);
    });
  }
}

window.PipelineVisualizer = PipelineVisualizer;
