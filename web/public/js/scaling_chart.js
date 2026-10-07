/**
 * scaling_chart.js - CPU vs GPU Empirical Performance & Scaling Lab
 * Canvas charts for grid scaling (1k-200k buses), GPU speedups, and residual descent.
 */

class ScalingChart {
  constructor(scalingCanvas, convergenceCanvas) {
    this.scalingCanvas = scalingCanvas;
    this.convCanvas = convergenceCanvas;
    this.scalingData = null;
    this.convergenceHistory = null;
  }

  setBenchmarkData(benchmarks) {
    if (benchmarks && benchmarks.e1_scaling) {
      this.scalingData = benchmarks.e1_scaling;
      this.renderScaling();
    }
  }

  setConvergenceHistory(history) {
    this.convergenceHistory = history;
    this.renderConvergence();
  }

  renderScaling() {
    if (!this.scalingCanvas || !this.scalingData) return;
    const canvas = this.scalingCanvas;
    const parent = canvas.parentElement;
    if (!parent || parent.clientWidth <= 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(100, parent.clientWidth - 24);
    const h = 200;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, w, h);

    const padLeft = 45;
    const padBottom = 30;
    const padTop = 15;
    const padRight = 20;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    // Grid sizes: 1000 to 200000 (log scale)
    const logMinX = Math.log10(1000);
    const logMaxX = Math.log10(200000);

    // Solve times: 0.01s to 100s (log scale)
    const logMinY = Math.log10(0.01);
    const logMaxY = Math.log10(120.0);

    const mapX = (val) => padLeft + ((Math.log10(val) - logMinX) / (logMaxX - logMinX)) * plotW;
    const mapY = (val) => padTop + plotH - ((Math.log10(Math.max(0.01, val)) - logMinY) / (logMaxY - logMinY)) * plotH;

    // Axes
    ctx.strokeStyle = 'rgba(70, 100, 150, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Axis Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9.5px JetBrains Mono';
    ctx.textAlign = 'center';
    [1000, 10000, 50000, 200000].forEach(val => {
      const x = mapX(val);
      ctx.fillText(val >= 1000 ? `${val / 1000}k` : val, x, padTop + plotH + 16);
    });

    ctx.textAlign = 'right';
    [0.01, 0.1, 1, 10, 100].forEach(val => {
      const y = mapY(val);
      ctx.fillText(`${val}s`, padLeft - 6, y + 3);
    });

    // Helper to draw series
    const drawSeries = (key, color, isDashed = false) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.fillStyle = color;
      ctx.setLineDash(isDashed ? [5, 4] : []);
      ctx.beginPath();

      this.scalingData.forEach((row, idx) => {
        const x = mapX(row.buses);
        const y = mapY(row[key]);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Points
      this.scalingData.forEach(row => {
        const x = mapX(row.buses);
        const y = mapY(row[key]);
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      });
    };

    drawSeries('cpu1_s', '#64748b', true); // CPU 1-thread (dashed gray)
    drawSeries('cpu4_s', '#94a3b8', true); // CPU 4-thread
    drawSeries('gpu_s', '#00e5ff', false); // GPU (solid cyan)

    // Legend
    ctx.setLineDash([]);
    ctx.font = '10px Inter';
    ctx.textAlign = 'left';

    ctx.fillStyle = '#00e5ff';
    ctx.fillText('■ GPU (CUDA / cuSPARSE)', padLeft + 10, padTop + 14);

    ctx.fillStyle = '#94a3b8';
    ctx.fillText('■ CPU 4-threads', padLeft + 170, padTop + 14);

    ctx.fillStyle = '#64748b';
    ctx.fillText('■ CPU 1-thread', padLeft + 280, padTop + 14);
  }

  renderConvergence() {
    if (!this.convCanvas || !this.convergenceHistory) return;
    const canvas = this.convCanvas;
    const parent = canvas.parentElement;
    if (!parent || parent.clientWidth <= 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(100, parent.clientWidth - 24);
    const h = 180;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, w, h);

    const padLeft = 55;
    const padBottom = 26;
    const padTop = 15;
    const padRight = 20;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    const hist = this.convergenceHistory;
    const iters = hist.length;

    // Log scale for mismatch: 1e-12 to 1e1
    const logMinY = -12;
    const logMaxY = 1;

    const mapX = (i) => padLeft + (i / Math.max(1, iters - 1)) * plotW;
    const mapY = (val) => {
      const logV = Math.log10(Math.max(1e-13, val));
      return padTop + plotH - ((logV - logMinY) / (logMaxY - logMinY)) * plotH;
    };

    // Axes
    ctx.strokeStyle = 'rgba(70, 100, 150, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9.5px JetBrains Mono';
    ctx.textAlign = 'right';
    [-12, -8, -4, 0].forEach(exp => {
      const y = padTop + plotH - ((exp - logMinY) / (logMaxY - logMinY)) * plotH;
      ctx.fillText(`1e${exp}`, padLeft - 6, y + 3);
    });

    ctx.textAlign = 'center';
    hist.forEach((_, idx) => {
      const x = mapX(idx);
      ctx.fillText(`It ${idx}`, x, padTop + plotH + 16);
    });

    // Tolerance line (1e-8)
    const tolY = mapY(1e-8);
    ctx.strokeStyle = 'rgba(255, 23, 68, 0.5)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padLeft, tolY);
    ctx.lineTo(padLeft + plotW, tolY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ff1744';
    ctx.textAlign = 'right';
    ctx.fillText('Tol 1e-8 pu', padLeft + plotW - 4, tolY - 5);

    // Mismatch curve
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    hist.forEach((pt, idx) => {
      const x = mapX(idx);
      const y = mapY(pt.max_mismatch_pu);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots
    ctx.fillStyle = '#ffffff';
    hist.forEach((pt, idx) => {
      const x = mapX(idx);
      const y = mapY(pt.max_mismatch_pu);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

window.ScalingChart = ScalingChart;
