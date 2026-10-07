#!/usr/bin/env python3
"""
pf_reference.py - Independent Newton-Raphson load-flow solver (NumPy/SciPy).

It shares no code with the CUDA program: it builds the bus admittance matrix
itself and solves every Newton step EXACTLY with SciPy's direct sparse solver
(SuperLU), not with an iterative method. A backtracking step (halve the step
while the mismatch would grow) keeps exact Newton stable from a flat start on
very large grids.

usage:
  python pf_reference.py <case file>                     solve and print a summary
  python pf_reference.py <case file> --published         compare with the Vm0/Va0 columns
  python pf_reference.py <case file> --compare <sol>     compare with a solution written by pf_cuda
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def read_case(path):
    lines = [ln.split('#')[0].split() for ln in open(path)]
    lines = [ln for ln in lines if ln]
    it = iter(lines)
    base = 100.0
    buses, gens, branches = [], [], []
    for ln in it:
        key = ln[0]
        if key == 'baseMVA':
            base = float(ln[1])
        elif key == 'buses':
            buses = [list(map(float, next(it))) for _ in range(int(ln[1]))]
        elif key == 'gens':
            gens = [list(map(float, next(it))) for _ in range(int(ln[1]))]
        elif key == 'branches':
            branches = [list(map(float, next(it))) for _ in range(int(ln[1]))]
    return base, np.array(buses), np.array(gens), np.array(branches)


def build(base, bus, gen, br):
    n = len(bus)
    ids = {int(b): k for k, b in enumerate(bus[:, 0])}
    btype = bus[:, 1].astype(int)
    f = np.array([ids[int(x)] for x in br[:, 0]])
    t = np.array([ids[int(x)] for x in br[:, 1]])
    ys = 1.0 / (br[:, 2] + 1j * br[:, 3])
    bc = br[:, 4]
    tap = np.where(br[:, 5] == 0, 1.0, br[:, 5])
    ytt = ys + 1j * bc / 2
    yff = ytt / (tap * tap)
    yft = -ys / tap
    ytf = -ys / tap
    ysh = (bus[:, 4] + 1j * bus[:, 5]) / base
    rows = np.concatenate([f, f, t, t, np.arange(n)])
    cols = np.concatenate([f, t, f, t, np.arange(n)])
    vals = np.concatenate([yff, yft, ytf, ytt, ysh])
    Y = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    Y.sum_duplicates()
    Sbus = -(bus[:, 2] + 1j * bus[:, 3]) / base
    V0 = np.ones(n, dtype=complex)
    for g in gen:
        k = ids[int(g[0])]
        Sbus[k] += g[1] / base
        V0[k] = g[2]
    return Y, Sbus, V0, btype


def newton(Y, Sbus, V0, btype, tol=1e-8, max_it=20, verbose=True):
    ref = np.where(btype == 3)[0]
    pv = np.where(btype == 2)[0]
    pq = np.where(btype == 1)[0]
    pvpq = np.concatenate([pv, pq])
    V = V0.copy()
    Va, Vm = np.angle(V), np.abs(V)
    history = []
    for k in range(max_it + 1):
        mis = V * np.conj(Y @ V) - Sbus
        F = np.concatenate([mis[pvpq].real, mis[pq].imag])
        norm = np.max(np.abs(F))
        history.append(norm)
        if verbose:
            print(f"  iteration {k}: max mismatch = {norm:.3e} pu")
        if norm < tol:
            return V, k, history
        Ibus = Y @ V
        dV = sp.diags(V)
        dS_dVm = dV @ np.conj(Y @ sp.diags(V / np.abs(V))) + np.conj(sp.diags(Ibus)) @ sp.diags(V / np.abs(V))
        dS_dVa = 1j * dV @ np.conj(sp.diags(Ibus) - Y @ dV)
        dS_dVm, dS_dVa = sp.csr_matrix(dS_dVm), sp.csr_matrix(dS_dVa)
        J = sp.bmat([[dS_dVa[pvpq][:, pvpq].real, dS_dVm[pvpq][:, pq].real],
                     [dS_dVa[pq][:, pvpq].imag, dS_dVm[pq][:, pq].imag]], format='csc')
        dx = -spla.spsolve(J, F)
        # damped Newton (backtracking): halve the exact step while it would increase max|F|
        step = 1.0
        for _ in range(12):
            Va_t, Vm_t = Va.copy(), Vm.copy()
            Va_t[pvpq] += step * dx[:len(pvpq)]
            Vm_t[pq] += step * dx[len(pvpq):]
            Vt = Vm_t * np.exp(1j * Va_t)
            mt = Vt * np.conj(Y @ Vt) - Sbus
            if max(np.max(np.abs(mt[pvpq].real)), np.max(np.abs(mt[pq].imag))) < norm or step < 1e-3:
                break
            step *= 0.5
        if verbose and step < 1.0:
            print(f"    (damped step: {step:g} x Newton step)")
        Va, Vm, V = Va_t, Vm_t, Vt
    raise RuntimeError('Newton-Raphson did not converge')


def main():
    path = sys.argv[1]
    base, bus, gen, br = read_case(path)
    Y, Sbus, V0, btype = build(base, bus, gen, br)
    print(f"Reference solver (SciPy SuperLU) on {path}: {len(bus)} buses, {len(br)} branches, "
          f"Ybus nnz = {Y.nnz}")
    t0 = time.perf_counter()
    V, its, _ = newton(Y, Sbus, V0, btype, verbose='--quiet' not in sys.argv)
    dt = time.perf_counter() - t0
    Vm, Va = np.abs(V), np.degrees(np.angle(V))
    print(f"  converged in {its} Newton iterations, {dt:.3f} s")
    ok = True
    if '--published' in sys.argv:
        dVm = np.max(np.abs(Vm - bus[:, 6]))
        dVa = np.max(np.abs(Va - bus[:, 7]))
        print(f"  vs published solution: max |dVm| = {dVm:.4f} pu, max |dVa| = {dVa:.3f} deg "
              f"(the published table is rounded to 3 and 2 decimals and came from another program)")
        ok = dVm < 0.002 and dVa < 0.02
        for k in range(len(bus)):
            print(f"    bus {int(bus[k,0]):2d}: Vm {Vm[k]:.4f} (pub {bus[k,6]:.3f})   "
                  f"Va {Va[k]:8.3f} (pub {bus[k,7]:7.2f})")
    if '--compare' in sys.argv:
        sol = sys.argv[sys.argv.index('--compare') + 1]
        data = np.loadtxt(sol)
        gVm, gVa = data[:, 1], data[:, 2]
        dVm = np.max(np.abs(Vm - gVm))
        dVa = np.max(np.abs(Va - gVa))
        # independent power-balance check of the CUDA solution itself
        Vg = gVm * np.exp(1j * np.radians(gVa))
        mis = Vg * np.conj(Y @ Vg) - Sbus
        pvpq = btype != 3
        pq = btype == 1
        mm = max(np.max(np.abs(mis[pvpq].real)), np.max(np.abs(mis[pq].imag)))
        print(f"  CUDA solution vs reference: max |dVm| = {dVm:.2e} pu, max |dVa| = {dVa:.2e} deg")
        print(f"  power-balance mismatch of the CUDA solution (recomputed here): {mm:.2e} pu")
        ok = dVm < 1e-6 and dVa < 1e-4 and mm < 1e-6
    print('  RESULT:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
