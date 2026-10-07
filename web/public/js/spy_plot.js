/**
 * spy_plot.js - Sparsity Spy Plot Matrix Visualizer
 * Visualizes CSR Ybus & Jacobian sparsity patterns and multicolour blocks.
 */

class SpyPlot {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.currentMatrix = null;
    this.matrixType = 'ybus'; // 'ybus' or 'jacobian'

    this.init();
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const parent = this.canvas.parentElement;
    if (!parent || parent.clientWidth <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(100, parent.clientWidth - 24);
    const h = 230;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render();
  }

  setData(ybusMatrix, jacobianMatrix) {
    this.ybusMatrix = ybusMatrix;
    this.jacobianMatrix = jacobianMatrix;
    this.currentMatrix = this.matrixType === 'ybus' ? this.ybusMatrix : this.jacobianMatrix;
    const parent = this.canvas.parentElement;
    if (parent && parent.clientWidth > 0) {
      this.resize();
    }
  }

  setMatrixType(type) {
    this.matrixType = type;
    this.currentMatrix = type === 'ybus' ? this.ybusMatrix : this.jacobianMatrix;
    this.render();
  }

  render() {
    const ctx = this.ctx;
    const w = parseFloat(this.canvas.style.width) || 300;
    const h = parseFloat(this.canvas.style.height) || 230;

    ctx.clearRect(0, 0, w, h);

    if (!this.currentMatrix || !this.currentMatrix.rowptr) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('Solve power flow to inspect sparse matrix structure', w / 2, h / 2);
      return;
    }

    const { n, nnz, rowptr, col } = this.currentMatrix;
    const padding = 28;
    const plotSize = Math.min(w - padding * 2, h - padding * 2);
    const startX = (w - plotSize) / 2;
    const startY = (h - plotSize) / 2;

    // Draw frame
    ctx.strokeStyle = 'rgba(70, 100, 150, 0.4)';
    ctx.lineWidth = 1;
    ctx.strokeRect(startX, startY, plotSize, plotSize);

    // Matrix Dimension labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`0`, startX, startY - 8);
    ctx.fillText(`${n}`, startX + plotSize, startY - 8);
    ctx.textAlign = 'right';
    ctx.fillText(`0`, startX - 8, startY + 10);
    ctx.fillText(`${n}`, startX - 8, startY + plotSize);

    // Draw non-zero entries (dots)
    const dotSize = Math.max(1.5, Math.min(6, plotSize / n));
    ctx.fillStyle = this.matrixType === 'ybus' ? '#00e5ff' : '#b388ff';

    for (let r = 0; r < n; r++) {
      const rowStart = rowptr[r];
      const rowEnd = rowptr[r + 1];
      const py = startY + (r / n) * plotSize;

      for (let k = rowStart; k < rowEnd; k++) {
        const c = col[k];
        const px = startX + (c / n) * plotSize;

        // Diagonal elements in white
        if (r === c) {
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(px, py, dotSize, dotSize);
          ctx.fillStyle = this.matrixType === 'ybus' ? '#00e5ff' : '#b388ff';
        } else {
          ctx.fillRect(px, py, dotSize, dotSize);
        }
      }
    }

    // Stats badge in bottom left
    const totalPossible = n * n;
    const sparsityPct = (100.0 * (1.0 - nnz / totalPossible)).toFixed(2);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'left';
    ctx.fillText(
      `${this.matrixType.toUpperCase()}: ${n}x${n} | nnz: ${nnz} | Sparsity: ${sparsityPct}%`,
      startX,
      startY + plotSize + 18
    );
  }
}

window.SpyPlot = SpyPlot;
