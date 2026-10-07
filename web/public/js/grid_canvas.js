/**
 * grid_canvas.js - Interactive Electrical Grid Topology Renderer
 * High-DPI HTML5 Canvas, Real-Time 5-Stage CUDA Pipeline Telemetry,
 * Live Current Flows, Mismatch Halos, Dynamic N-1 Line Outage Toggles.
 * Frontend visualization engine replicating the clean SaaS reference dashboard.
 */

class GridCanvas {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.grid = null;
    this.solution = null;

    // Viewport transform (Pan & Zoom)
    this.transform = { x: 30, y: 30, scale: 1.0 };
    this.targetTransform = { x: 30, y: 30, scale: 1.0 };

    // Interaction states
    this.isPanning = false;
    this.draggedBus = null;
    this.hoveredBus = null;
    this.hoveredBranch = null;
    this.lastMouse = { x: 0, y: 0 };

    // Live CUDA pipeline stage states
    this.simState = {
      stage: 0,
      iteration: 0,
      isFlowing: false,
      busMismatches: [],
      lineFlows: [],
      outageLine: -1,
      isConverged: false,
      voltages: null
    };

    this.dashTimer = 0;
    this.animFrameId = null;

    // Event callbacks
    this.onBusClick = options.onBusClick || null;
    this.onBranchClick = options.onBranchClick || null;

    this.initCanvas();
    this.setupEventListeners();
    this.startAnimationLoop();
  }

  initCanvas() {
    this.resize();
    window.addEventListener('resize', () => this.resize());
    if (window.ResizeObserver && this.canvas.parentElement) {
      let rAF = null;
      const ro = new ResizeObserver(() => {
        if (rAF) cancelAnimationFrame(rAF);
        rAF = requestAnimationFrame(() => this.resize());
      });
      ro.observe(this.canvas.parentElement);
    }
  }

  resize() {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const dpr = window.devicePixelRatio || 1;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.fitToScreen();
    this.render();
  }

  setGrid(grid, solution = null) {
    this.grid = grid;
    this.solution = solution;
    this.fitToScreen();
    this.render();
  }

  setSolution(solution) {
    this.solution = solution;
    this.render();
  }

  setSimulationState(state) {
    this.simState = Object.assign(this.simState, state);
    this.render();
  }

  setupEventListeners() {
    this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
    window.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    window.addEventListener('mouseup', () => this.handleMouseUp());
    this.canvas.addEventListener('wheel', (e) => this.handleWheel(e), { passive: false });
  }

  screenToWorld(sx, sy) {
    const rect = this.canvas.getBoundingClientRect();
    const x = (sx - rect.left - this.transform.x) / this.transform.scale;
    const y = (sy - rect.top - this.transform.y) / this.transform.scale;
    return { x, y };
  }

  handleMouseDown(e) {
    const pos = this.screenToWorld(e.clientX, e.clientY);
    const clickedBus = this.findBusAt(pos.x, pos.y);

    if (clickedBus) {
      if (e.button === 0) {
        this.draggedBus = clickedBus;
        this.lastMouse = { x: e.clientX, y: e.clientY };
      }
    } else {
      const clickedBranch = this.findBranchAt(pos.x, pos.y);
      if (clickedBranch) {
        if (this.onBranchClick) this.onBranchClick(clickedBranch);
      } else {
        this.isPanning = true;
        this.lastMouse = { x: e.clientX, y: e.clientY };
      }
    }
  }

  handleMouseMove(e) {
    const pos = this.screenToWorld(e.clientX, e.clientY);

    if (this.draggedBus) {
      const dx = (e.clientX - this.lastMouse.x) / this.transform.scale;
      const dy = (e.clientY - this.lastMouse.y) / this.transform.scale;
      this.draggedBus.x += dx;
      this.draggedBus.y += dy;
      this.lastMouse = { x: e.clientX, y: e.clientY };
      this.render();
      return;
    }

    if (this.isPanning) {
      const dx = e.clientX - this.lastMouse.x;
      const dy = e.clientY - this.lastMouse.y;
      this.transform.x += dx;
      this.transform.y += dy;
      this.targetTransform.x = this.transform.x;
      this.targetTransform.y = this.transform.y;
      this.lastMouse = { x: e.clientX, y: e.clientY };
      this.render();
      return;
    }

    const bus = this.findBusAt(pos.x, pos.y);
    const branch = bus ? null : this.findBranchAt(pos.x, pos.y);

    let changed = false;
    if (this.hoveredBus !== bus) {
      this.hoveredBus = bus;
      changed = true;
    }
    if (this.hoveredBranch !== branch) {
      this.hoveredBranch = branch;
      changed = true;
    }

    this.canvas.style.cursor = bus ? 'pointer' : (branch ? 'pointer' : 'default');
    if (changed) this.render();
  }

  handleMouseUp() {
    if (this.draggedBus) {
      if (this.onBusClick) this.onBusClick(this.draggedBus);
      this.draggedBus = null;
    }
    this.isPanning = false;
  }

  handleWheel(e) {
    e.preventDefault();
    const parent = this.canvas.parentElement;
    const w = parent ? parent.clientWidth : 800;
    const h = parent ? parent.clientHeight : 600;
    const cx = w / 2;
    const cy = h / 2;

    const delta = Math.max(-60, Math.min(60, e.deltaY));
    const factor = Math.exp(-delta * 0.0035);
    const newScale = Math.max(0.4, Math.min(3.0, this.targetTransform.scale * factor));

    this.targetTransform.x = cx - (cx - this.targetTransform.x) * (newScale / this.targetTransform.scale);
    this.targetTransform.y = cy - (cy - this.targetTransform.y) * (newScale / this.targetTransform.scale);
    this.targetTransform.scale = newScale;
  }

  zoomCentral(factor) {
    const parent = this.canvas.parentElement;
    const w = parent ? parent.clientWidth : 800;
    const h = parent ? parent.clientHeight : 600;
    const cx = w / 2;
    const cy = h / 2;

    const newScale = Math.max(0.4, Math.min(3.0, this.targetTransform.scale * factor));
    this.targetTransform.x = cx - (cx - this.targetTransform.x) * (newScale / this.targetTransform.scale);
    this.targetTransform.y = cy - (cy - this.targetTransform.y) * (newScale / this.targetTransform.scale);
    this.targetTransform.scale = newScale;
  }

  findBusAt(x, y) {
    if (!this.grid || !this.grid.buses) return null;
    const radius = 22;
    for (const b of this.grid.buses) {
      const dx = b.x - x;
      const dy = b.y - y;
      if (dx * dx + dy * dy <= radius * radius) {
        return b;
      }
    }
    return null;
  }

  findBranchAt(x, y) {
    if (!this.grid || !this.grid.branches) return null;
    const idMap = new Map();
    this.grid.buses.forEach(b => idMap.set(b.id, b));

    for (let k = 0; k < this.grid.branches.length; k++) {
      const br = this.grid.branches[k];
      const fb = idMap.get(br.from);
      const tb = idMap.get(br.to);
      if (!fb || !tb) continue;

      const dist = this.pointToSegmentDistance(x, y, fb.x, fb.y, tb.x, tb.y);
      if (dist < 12) {
        return { branch: br, index: k };
      }
    }
    return null;
  }

  pointToSegmentDistance(px, py, x1, y1, x2, y2) {
    const l2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    if (l2 === 0) return Math.hypot(px - x1, py - y1);
    let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (x1 + t * (x2 - x1)), py - (y1 + t * (y2 - y1)));
  }

  fitToScreen() {
    if (!this.grid || !this.grid.buses || this.grid.buses.length === 0) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    this.grid.buses.forEach(b => {
      minX = Math.min(minX, b.x);
      maxX = Math.max(maxX, b.x);
      minY = Math.min(minY, b.y);
      maxY = Math.max(maxY, b.y);
    });

    const parent = this.canvas.parentElement;
    const w = parent ? parent.clientWidth : 800;
    const h = parent ? parent.clientHeight : 450;

    // Account for node radius (18px) and label offsets (38px below node)
    const effectiveMinX = minX - 32;
    const effectiveMaxX = maxX + 32;
    const effectiveMinY = minY - 28;
    const effectiveMaxY = maxY + 42;

    const gridW = effectiveMaxX - effectiveMinX;
    const gridH = effectiveMaxY - effectiveMinY;
    const pad = 18; // clean breathing margin inside canvas

    const scale = Math.min(1.25, Math.max(0.35, Math.min((w - pad * 2) / gridW, (h - pad * 2) / gridH)));
    const gridCenterX = (effectiveMinX + effectiveMaxX) / 2;
    const gridCenterY = (effectiveMinY + effectiveMaxY) / 2;

    const tx = w / 2 - gridCenterX * scale;
    const ty = h / 2 - gridCenterY * scale;

    this.targetTransform.scale = scale;
    this.targetTransform.x = tx;
    this.targetTransform.y = ty;

    this.transform.scale = scale;
    this.transform.x = tx;
    this.transform.y = ty;
    this.render();
  }

  startAnimationLoop() {
    const loop = () => {
      this.dashTimer += 0.05;

      // Smooth camera transform lerp
      const lerp = 0.22;
      const dScale = Math.abs(this.targetTransform.scale - this.transform.scale);
      const dX = Math.abs(this.targetTransform.x - this.transform.x);
      const dY = Math.abs(this.targetTransform.y - this.transform.y);

      if (dScale > 0.0005 || dX > 0.05 || dY > 0.05) {
        this.transform.scale += (this.targetTransform.scale - this.transform.scale) * lerp;
        this.transform.x += (this.targetTransform.x - this.transform.x) * lerp;
        this.transform.y += (this.targetTransform.y - this.transform.y) * lerp;
        this.render();
      } else if (this.simState.isFlowing || this.simState.stage === 1) {
        // Continuous dash animation while SpMV is flowing
        this.render();
      }

      this.animFrameId = requestAnimationFrame(loop);
    };
    this.animFrameId = requestAnimationFrame(loop);
  }

  render() {
    const ctx = this.ctx;
    const parent = this.canvas.parentElement;
    const w = parent ? parent.clientWidth : 800;
    const h = parent ? parent.clientHeight : 600;

    ctx.save();
    ctx.clearRect(0, 0, w, h);

    // Pristine white background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    if (!this.grid || !this.grid.buses) {
      ctx.restore();
      return;
    }

    ctx.translate(this.transform.x, this.transform.y);
    ctx.scale(this.transform.scale, this.transform.scale);

    const idMap = new Map();
    this.grid.buses.forEach(b => idMap.set(b.id, b));

    const isSpMVStage = this.simState.stage === 1 || this.simState.isFlowing;
    const isConverged = this.simState.isConverged;
    const flows = this.simState.lineFlows || [];
    const outageLine = this.simState.outageLine !== undefined ? this.simState.outageLine : -1;

    // 1. Draw Transmission Lines
    this.grid.branches.forEach((br, k) => {
      const fb = idMap.get(br.from);
      const tb = idMap.get(br.to);
      if (!fb || !tb) return;

      const isOutage = k === outageLine || br.status === 0;
      const isHovered = this.hoveredBranch && this.hoveredBranch.index === k;
      const flowMw = flows[k] !== undefined ? flows[k] : 0;

      ctx.save();
      if (isOutage) {
        // Tripped line: Red dashed with TRIPPED badge
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2.0;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(fb.x, fb.y);
        ctx.lineTo(tb.x, tb.y);
        ctx.stroke();

        const mx = (fb.x + tb.x) / 2;
        const my = (fb.y + tb.y) / 2;
        ctx.fillStyle = '#ef4444';
        ctx.font = '600 10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('TRIPPED', mx, my - 8);
      } else if (isSpMVStage) {
        // Stage 2 (SpMV): Animated flowing current dashes!
        ctx.strokeStyle = isHovered ? '#1d4ed8' : '#2563eb';
        ctx.lineWidth = isHovered ? 3.0 : 2.2;
        ctx.setLineDash([6, 5]);
        ctx.lineDashOffset = -this.dashTimer * 16;
        ctx.beginPath();
        ctx.moveTo(fb.x, fb.y);
        ctx.lineTo(tb.x, tb.y);
        ctx.stroke();
      } else {
        // Normal active line: clean slate line
        ctx.strokeStyle = isHovered ? '#475569' : '#94a3b8';
        ctx.lineWidth = isHovered ? 2.5 : 1.8;
        ctx.beginPath();
        ctx.moveTo(fb.x, fb.y);
        ctx.lineTo(tb.x, tb.y);
        ctx.stroke();

        // If converged, draw directional power flow triangle arrow and MW flow
        if (isConverged && Math.abs(flowMw) >= 1) {
          const flowFromTo = flowMw >= 0;
          const dx = flowFromTo ? (tb.x - fb.x) : (fb.x - tb.x);
          const dy = flowFromTo ? (tb.y - fb.y) : (fb.y - tb.y);
          const angle = Math.atan2(dy, dx);
          const mx = (fb.x + tb.x) / 2;
          const my = (fb.y + tb.y) / 2;

          ctx.save();
          ctx.translate(mx, my);
          ctx.rotate(angle);
          ctx.beginPath();
          ctx.moveTo(6, 0);
          ctx.lineTo(-4, -4);
          ctx.lineTo(-4, 4);
          ctx.closePath();
          ctx.fillStyle = isHovered ? '#1d4ed8' : '#2563eb';
          ctx.fill();
          ctx.restore();

          // Flow label in MW
          ctx.fillStyle = isHovered ? '#1d4ed8' : '#475569';
          ctx.font = '600 10px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText(`${Math.abs(flowMw).toFixed(0)} MW`, mx, my - 6);
        }
      }
      ctx.restore();
    });

    // 2. Draw Buses (Solid Circles + White ID + Voltage Below + Mismatch Ring)
    const mismatches = this.simState.busMismatches || [];
    const voltages = this.simState.voltages || [];

    this.grid.buses.forEach((b, idx) => {
      const isHovered = this.hoveredBus === b;
      const nodeRadius = isHovered ? 18 : 16;

      // Color coding: Slack = Blue, Generator = Green, Load = Orange
      let baseColor = '#f97316'; // Load (Orange)
      if (b.type === 3) {
        baseColor = '#2563eb'; // Slack (Blue)
      } else if (b.type === 2 || (this.grid.gens && this.grid.gens.some(g => g.bus === b.id))) {
        baseColor = '#10b981'; // Generator (Green)
      }

      ctx.save();

      // Stage 3 (Mismatch): Red pulsing mismatch ring around buses with high mismatch
      const m = mismatches[idx] || 0;
      if (this.simState.stage >= 2 && m > 1e-8) {
        const ringExtra = 10 * Math.max(0, Math.min(1, (Math.log10(m) + 8) / 8));
        ctx.beginPath();
        ctx.arc(b.x, b.y, nodeRadius + 3 + ringExtra, 0, Math.PI * 2);
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2.4;
        ctx.stroke();
      }

      // Hover aura
      if (isHovered) {
        ctx.beginPath();
        ctx.arc(b.x, b.y, nodeRadius + 5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(15, 23, 42, 0.08)';
        ctx.fill();
      }

      // Solid Node Circle
      ctx.beginPath();
      ctx.arc(b.x, b.y, nodeRadius, 0, Math.PI * 2);
      ctx.fillStyle = baseColor;
      ctx.fill();

      // Bus Number (bold white centered inside)
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px Inter, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${b.id}`, b.x, b.y);

      // Voltage text beneath node (e.g. "1.060 pu")
      const vmVal = voltages[idx] !== undefined ? voltages[idx] : (b.Vm0 || 1.0);
      ctx.fillStyle = '#475569';
      ctx.font = '500 11px Inter, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`${vmVal.toFixed(3)} pu`, b.x, b.y + nodeRadius + 6);

      ctx.restore();
    });

    ctx.restore();
  }
}

window.GridCanvas = GridCanvas;
