/**
 * contingency_view.js - N-1 Contingency Analysis & Screening Dashboard
 * Displays ranked security indices, voltage violations, and outage injectors.
 */

class ContingencyView {
  constructor(tableContainerElement, onApplyOutageCallback = null) {
    this.container = tableContainerElement;
    this.onApplyOutage = onApplyOutageCallback;
    this.report = null;
    this.filterOnlyViolations = false;
  }

  setReport(report) {
    this.report = report;
    this.render();
  }

  toggleFilter(onlyViolations) {
    this.filterOnlyViolations = onlyViolations;
    this.render();
  }

  render() {
    if (!this.container) return;
    this.container.innerHTML = '';

    if (!this.report || !this.report.contingencies) {
      this.container.innerHTML = `
        <div style="padding: 24px; text-align: center; color: var(--text-dim); font-size: 12px;">
          Click "Run N-1 Contingency Screening" to remotely evaluate transmission line outages.
        </div>
      `;
      return;
    }

    const filtered = this.report.contingencies.filter(c => {
      if (this.filterOnlyViolations) {
        return c.severity === 'CRITICAL' || c.severity === 'WARNING';
      }
      return true;
    });

    const table = document.createElement('table');
    table.className = 'data-table';

    table.innerHTML = `
      <thead>
        <tr>
          <th>Branch</th>
          <th>Severity</th>
          <th>PI Index</th>
          <th>Max ΔV</th>
          <th>Worst Bus</th>
          <th>Overloads</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody></tbody>
    `;

    const tbody = table.querySelector('tbody');

    filtered.forEach(c => {
      const tr = document.createElement('tr');

      let sevClass = 'normal';
      if (c.severity === 'CRITICAL') sevClass = 'critical';
      else if (c.severity === 'WARNING') sevClass = 'warning';

      tr.innerHTML = `
        <td style="font-weight: 600; color: #f1f5f9;">${c.branchName}</td>
        <td><span class="status-tag ${sevClass}">${c.severity}</span></td>
        <td>${c.performanceIndex > 10000 ? 'COLLAPSE' : c.performanceIndex.toFixed(1)}</td>
        <td>${c.maxVoltageDeviation ? c.maxVoltageDeviation.toFixed(3) : '-'}</td>
        <td>${c.worstBus ? `Bus ${c.worstBus}` : '-'}</td>
        <td>${c.overloadedBranches}</td>
        <td>
          <button class="btn btn-secondary" style="padding: 3px 8px; font-size: 10px;" data-branch-idx="${c.branchIndex}">
            Apply Outage
          </button>
        </td>
      `;

      const btn = tr.querySelector('button');
      btn.addEventListener('click', () => {
        if (this.onApplyOutage) this.onApplyOutage(c.branchIndex);
      });

      tbody.appendChild(tr);
    });

    this.container.appendChild(table);
  }
}

window.ContingencyView = ContingencyView;
