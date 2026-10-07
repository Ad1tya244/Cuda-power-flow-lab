// =============================================================================
//  powerflow.h  -  Newton-Raphson load flow: grid model + shared maths + CPU solver
//
//  Everything that computes a NUMBER is a __host__ __device__ function, so the
//  CPU baseline and the CUDA kernels run exactly the same formulas.
//
//  Notation (standard power-systems per-unit notation)
//    Ybus   : bus admittance matrix, complex, SPARSE (~4 non-zeros per row)
//    V      : complex bus voltages  V_i = Vm_i * e^(j*Va_i)
//    I      : injected currents     I = Ybus * V          (sparse mat-vec)
//    S      : injected power        S_i = V_i * conj(I_i)
//    F      : mismatch  [ dP(pv+pq) ; dQ(pq) ]  = S(V) - S_specified
//    J      : Jacobian  dF/dx, x = [ Va(pv+pq) ; Vm(pq) ]  (same sparsity as Ybus)
//    Newton : J * dx = -F ,  x <- x + dx , until max|F| < 1e-8 pu
// =============================================================================
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#define ALIGN16 __align__(16)
#else
#define HD inline
#define ALIGN16 alignas(16)
#endif

// ---------------------------------------------------------------------------
//  Complex numbers (layout-compatible with cuDoubleComplex for cuSPARSE)
// ---------------------------------------------------------------------------
struct ALIGN16 cplx { double re, im; };
HD cplx mk(double r, double i) { cplx c; c.re = r; c.im = i; return c; }
HD cplx cadd(cplx a, cplx b) { return mk(a.re + b.re, a.im + b.im); }
HD cplx csub(cplx a, cplx b) { return mk(a.re - b.re, a.im - b.im); }
HD cplx cmul(cplx a, cplx b) { return mk(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
HD cplx cconj(cplx a) { return mk(a.re, -a.im); }
HD cplx cscale(cplx a, double s) { return mk(a.re * s, a.im * s); }
HD cplx cdivr(double num, cplx d) {                     // num / d
    const double den = d.re * d.re + d.im * d.im;
    return mk(num * d.re / den, -num * d.im / den);
}

// ---------------------------------------------------------------------------
//  Per-element maths shared by CPU and GPU
// ---------------------------------------------------------------------------
// Jacobian entry kinds:  0 = dP/dVa, 1 = dP/dVm, 2 = dQ/dVa, 3 = dQ/dVm
// Formulas (polar form, as in MATPOWER's dSbus_dV):
//   dS_i/dVa_j = j * V_i * conj(delta_ij * I_i - Y_ij * V_j)
//   dS_i/dVm_j = V_i * conj(Y_ij * V_j/|V_j|) + delta_ij * conj(I_i) * V_i/|V_i|
HD double jacobian_entry(int kind, int i, int j, cplx Yij, const cplx* V, const double* Vm,
                         const cplx* I) {
    const cplx Vi = V[i];
    if (kind == 0 || kind == 2) {
        cplx t = cmul(Yij, V[j]);
        t = (i == j) ? csub(I[i], t) : mk(-t.re, -t.im);
        const cplx s = cmul(Vi, cconj(t));
        const cplx js = mk(-s.im, s.re);               // multiply by j
        return kind == 0 ? js.re : js.im;
    }
    cplx s = cmul(Vi, cconj(cmul(Yij, cscale(V[j], 1.0 / Vm[j]))));
    if (i == j) s = cadd(s, cmul(cconj(I[i]), cscale(Vi, 1.0 / Vm[i])));
    return kind == 1 ? s.re : s.im;
}

// Mismatch of one equation: active power (isQ = 0) or reactive power (isQ = 1) at bus i
HD double mismatch_entry(int isQ, int i, const cplx* V, const cplx* I, const double* Psp, const double* Qsp) {
    const cplx S = cmul(V[i], cconj(I[i]));
    return isQ ? S.im - Qsp[i] : S.re - Psp[i];
}

// ---------------------------------------------------------------------------
//  Grid model
// ---------------------------------------------------------------------------
struct Branch { int f, t; double r, x, b, tap; };

struct Grid {
    std::string name;
    int n = 0;
    double base = 100.0;
    std::vector<int> type;                 // 1 = PQ (load), 2 = PV (generator), 3 = slack
    std::vector<double> Pd, Qd, Gs, Bs;    // MW / MVAr
    std::vector<double> Pg, Vset;          // generator output (MW) and voltage set-point (pu)
    std::vector<double> VmPub, VaPub;      // published solution (IEEE cases only)
    std::vector<Branch> br;
    bool has_published = false;
};

struct Csr {                                // compressed sparse row, 0-based, sorted columns
    int n = 0, nnz = 0;
    std::vector<int> rowptr, col;
};

enum class Ordering { NATURAL, MULTICOLOR };

// Complete problem ready for Newton-Raphson (built once from a Grid)
struct Problem {
    int n = 0, npvpq = 0, npq = 0, nx = 0;
    Csr Y;                                  // Ybus pattern
    std::vector<cplx> Yval;                 // Ybus values
    std::vector<double> Psp, Qsp;           // specified injections (pu)
    std::vector<double> Vm0, Va0;           // start point (flat start + generator set-points)
    std::vector<int> rowbus;                // bus of each equation row (size nx)
    std::vector<int> rowq;                  // 0 = active-power row, 1 = reactive-power row
    int colors = 1, levels = 0;             // bus colours used, triangular-solve levels
    // multicolour ordering only: non-slack buses listed colour by colour; bus k owns
    // equation rows brow0[k] .. brow0[k]+bcnt[k]-1; colour c = buses cstart[c]..cstart[c+1]-1
    std::vector<int> cstart, brow0, bcnt;
    std::vector<int> jdiag;                 // position of the diagonal entry of each Jacobian row
    std::vector<int> xbus, xkind;           // unknown r updates Va (0) or Vm (1) of bus xbus[r]
    Csr J;                                  // Jacobian pattern
    std::vector<int> jy, jkind, jrow;       // per Jacobian non-zero: Ybus index, kind, row bus
    long long branches = 0;
};

static void die(const char* msg) { fprintf(stderr, "error: %s\n", msg); exit(EXIT_FAILURE); }

// --------------------------------------------------------------- case files
static Grid read_case(const std::string& path) {
    std::ifstream in(path);
    if (!in) die(("cannot open case file " + path).c_str());
    Grid g;
    g.name = path;
    std::string line;
    std::vector<std::vector<double>> rows;
    auto next_nums = [&](std::vector<double>& v) {
        while (std::getline(in, line)) {
            const auto h = line.find('#');
            if (h != std::string::npos) line = line.substr(0, h);
            std::istringstream ss(line);
            v.clear();
            double x;
            while (ss >> x) v.push_back(x);
            if (!v.empty()) return true;
        }
        return false;
    };
    std::vector<int> ids;
    while (std::getline(in, line)) {
        const auto h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key)) continue;
        int cnt = 0;
        if (key == "baseMVA") { ss >> g.base; continue; }
        ss >> cnt;
        std::vector<double> v;
        if (key == "buses") {
            g.n = cnt;
            g.type.resize(cnt); g.Pd.resize(cnt); g.Qd.resize(cnt); g.Gs.resize(cnt); g.Bs.resize(cnt);
            g.Pg.assign(cnt, 0.0); g.Vset.assign(cnt, 1.0); g.VmPub.resize(cnt); g.VaPub.resize(cnt);
            ids.resize(cnt);
            for (int k = 0; k < cnt; ++k) {
                if (!next_nums(v) || v.size() < 8) die("bad bus line");
                ids[k] = (int)v[0]; g.type[k] = (int)v[1]; g.Pd[k] = v[2]; g.Qd[k] = v[3];
                g.Gs[k] = v[4]; g.Bs[k] = v[5]; g.VmPub[k] = v[6]; g.VaPub[k] = v[7];
            }
            g.has_published = true;
        } else if (key == "gens") {
            for (int k = 0; k < cnt; ++k) {
                if (!next_nums(v) || v.size() < 3) die("bad generator line");
                const int b = (int)(std::find(ids.begin(), ids.end(), (int)v[0]) - ids.begin());
                g.Pg[b] += v[1]; g.Vset[b] = v[2];
            }
        } else if (key == "branches") {
            for (int k = 0; k < cnt; ++k) {
                if (!next_nums(v) || v.size() < 6) die("bad branch line");
                Branch b;
                b.f = (int)(std::find(ids.begin(), ids.end(), (int)v[0]) - ids.begin());
                b.t = (int)(std::find(ids.begin(), ids.end(), (int)v[1]) - ids.begin());
                b.r = v[2]; b.x = v[3]; b.b = v[4]; b.tap = v[5];
                g.br.push_back(b);
            }
        }
    }
    return g;
}

static void write_case(const Grid& g, const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) die(("cannot write " + path).c_str());
    fprintf(f, "# synthetic grid written by pf_cuda (%d buses)\nbaseMVA %.1f\nbuses %d\n", g.n, g.base, g.n);
    for (int i = 0; i < g.n; ++i)
        fprintf(f, "%d %d %.10g %.10g %.10g %.10g 1 0\n", i + 1, g.type[i], g.Pd[i], g.Qd[i], g.Gs[i], g.Bs[i]);
    int ng = 0;
    for (int i = 0; i < g.n; ++i) ng += (g.type[i] != 1);
    fprintf(f, "gens %d\n", ng);
    for (int i = 0; i < g.n; ++i)
        if (g.type[i] != 1) fprintf(f, "%d %.10g %.10g\n", i + 1, g.Pg[i], g.Vset[i]);
    fprintf(f, "branches %zu\n", g.br.size());
    for (const Branch& b : g.br)
        fprintf(f, "%d %d %.10g %.10g %.10g %.10g\n", b.f + 1, b.t + 1, b.r, b.x, b.b, b.tap);
    fclose(f);
}

// ------------------------------------------------------- synthetic grids
// A meshed "state-wide" transmission grid: buses on a sqrt(n) x sqrt(n) map,
// every bus wired to its east neighbour, about half of them to the south
// neighbour (column 0 always, so the grid is connected), plus 5% longer lines
// to buses up to 3 steps away. Average degree ~3, like real transmission grids.
// Power plants (wind, solar, thermal: PV buses) sit at every third bus in an even
// pattern, the slack plant in the middle of the map, every bus carries a load,
// and each plant is dispatched to cover the loads nearest to it.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni(double a, double b) { return a + (b - a) * ((next() >> 11) * (1.0 / 9007199254740992.0)); }
};

static Grid synthetic_grid(int n, uint64_t seed = 2026) {
    Grid g;
    g.name = "synthetic-" + std::to_string(n);
    g.n = n;
    g.type.assign(n, 1); g.Pd.resize(n); g.Qd.resize(n); g.Gs.assign(n, 0); g.Bs.assign(n, 0);
    g.Pg.assign(n, 0); g.Vset.assign(n, 1.0);
    Rng rng(seed + (uint64_t)n);
    const int s = (int)std::ceil(std::sqrt((double)n));
    auto add_line = [&](int a, int b) {
        Branch br;
        br.f = a; br.t = b;
        br.x = rng.uni(0.02, 0.06);                   // reactance, pu
        br.r = br.x / rng.uni(8.0, 15.0);             // high-voltage lines: X/R between 8 and 15
        br.b = rng.uni(0.01, 0.03);                   // line charging, pu
        br.tap = 0;
        g.br.push_back(br);
    };
    for (int i = 0; i < n; ++i) {
        const int x = i % s;
        if (x + 1 < s && i + 1 < n) add_line(i, i + 1);
        if (i + s < n && (x == 0 || rng.uni(0, 1) < 0.5)) add_line(i, i + s);
    }
    const int extra = n / 20;
    for (int k = 0; k < extra; ++k) {
        const int i = (int)(rng.next() % (uint64_t)n);
        const int dx = (int)(rng.next() % 7) - 3, dy = (int)(rng.next() % 7) - 3;
        const int x = i % s + dx, y = i / s + dy;
        if (x < 0 || x >= s || y < 0) continue;
        const int j = y * s + x;
        if (j >= n || j == i) continue;
        add_line(std::min(i, j), std::max(i, j));
    }
    for (int i = 0; i < n; ++i) {
        g.Pd[i] = rng.uni(2.0, 10.0);                 // MW
        g.Qd[i] = g.Pd[i] * rng.uni(0.2, 0.4);        // MVAr
    }
    // generators: the slack plant in the middle of the map, PV plants at every third
    // bus of the map (x + 2y divisible by 3), so every load has a plant close by
    const int slack = std::min(n - 1, (s / 2) * s + s / 2);
    std::vector<int> src;
    g.type[slack] = 3; g.Vset[slack] = 1.02;
    src.push_back(slack);
    for (int i = 0; i < n; ++i)
        if (i != slack && (i % s + 2 * (i / s)) % 3 == 0) {
            g.type[i] = 2;
            g.Vset[i] = rng.uni(1.015, 1.025);   // neighbouring plants hold coordinated voltage set-points
            src.push_back(i);
        }
    // regional dispatch: every load is served by its nearest generator (multi-source
    // breadth-first search over the lines), and each generator is dispatched to cover
    // its region, as operators do. The slack bus then only supplies the small losses.
    std::vector<std::vector<int>> adj(n);
    for (const Branch& b : g.br) { adj[b.f].push_back(b.t); adj[b.t].push_back(b.f); }
    std::vector<int> owner(n, -1), queue(src);
    for (int s0 : src) owner[s0] = s0;
    for (size_t h = 0; h < queue.size(); ++h)
        for (int j : adj[queue[h]])
            if (owner[j] < 0) { owner[j] = owner[queue[h]]; queue.push_back(j); }
    for (int i = 0; i < n; ++i) {
        if (owner[i] < 0) die("synthetic grid is not connected");
        if (owner[i] != slack) g.Pg[owner[i]] += g.Pd[i];
    }
    return g;
}

// ---------------------------------------------------------- build problem
static Problem build_problem(const Grid& g, Ordering ordering = Ordering::MULTICOLOR) {
    Problem P;
    const int n = g.n;
    P.n = n;
    P.branches = (long long)g.br.size();
    // --- Ybus from branch pi-models (triplets, then sort + merge into CSR)
    struct T { int r, c; cplx v; };
    std::vector<T> t;
    t.reserve(4 * g.br.size() + n);
    for (const Branch& b : g.br) {
        const cplx ys = cdivr(1.0, mk(b.r, b.x));
        const double tap = (b.tap == 0.0) ? 1.0 : b.tap;
        const cplx ytt = cadd(ys, mk(0, b.b / 2));
        const cplx yff = cscale(ytt, 1.0 / (tap * tap));
        const cplx yft = cscale(ys, -1.0 / tap);
        t.push_back({b.f, b.f, yff}); t.push_back({b.f, b.t, yft});
        t.push_back({b.t, b.f, yft}); t.push_back({b.t, b.t, ytt});
    }
    for (int i = 0; i < n; ++i) t.push_back({i, i, mk(g.Gs[i] / g.base, g.Bs[i] / g.base)});
    std::sort(t.begin(), t.end(), [](const T& a, const T& b) { return a.r != b.r ? a.r < b.r : a.c < b.c; });
    P.Y.n = n;
    P.Y.rowptr.assign(n + 1, 0);
    for (size_t k = 0; k < t.size();) {
        size_t e = k;
        cplx v = mk(0, 0);
        while (e < t.size() && t[e].r == t[k].r && t[e].c == t[k].c) v = cadd(v, t[e++].v);
        P.Y.col.push_back(t[k].c);
        P.Yval.push_back(v);
        P.Y.rowptr[t[k].r + 1]++;
        k = e;
    }
    for (int i = 0; i < n; ++i) P.Y.rowptr[i + 1] += P.Y.rowptr[i];
    P.Y.nnz = (int)P.Y.col.size();
    // --- specified injections and start point
    P.Psp.resize(n); P.Qsp.resize(n); P.Vm0.assign(n, 1.0); P.Va0.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        P.Psp[i] = (g.Pg[i] - g.Pd[i]) / g.base;
        P.Qsp[i] = -g.Qd[i] / g.base;
        if (g.type[i] != 1) P.Vm0[i] = g.Vset[i];
    }
    // --- unknowns: Va of every non-slack bus and Vm of every PQ bus.
    //   NATURAL    : all Va in bus order, then all Vm (the textbook/MATPOWER layout)
    //   MULTICOLOR : buses coloured so that no two connected buses share a colour,
    //                unknowns numbered colour by colour, (Va_i, Vm_i) kept together.
    //                Buses of one colour never touch each other, so the ILU(0)
    //                triangular solves need only ~2 steps per colour -> GPU friendly.
    for (int i = 0; i < n; ++i) { P.npvpq += (g.type[i] != 3); P.npq += (g.type[i] == 1); }
    P.nx = P.npvpq + P.npq;
    std::vector<int> idxVa(n, -1), idxVm(n, -1);
    int next = 0;
    if (ordering == Ordering::NATURAL) {
        P.colors = 1;
        for (int i = 0; i < n; ++i) if (g.type[i] != 3) idxVa[i] = next++;
        for (int i = 0; i < n; ++i) if (g.type[i] == 1) idxVm[i] = next++;
    } else {
        std::vector<int> color(n, -1), mark;
        for (int i = 0; i < n; ++i) {                  // greedy colouring of the bus graph
            mark.assign(mark.size(), 0);
            for (int k = P.Y.rowptr[i]; k < P.Y.rowptr[i + 1]; ++k) {
                const int c = color[P.Y.col[k]];
                if (c >= 0) { if ((int)mark.size() <= c) mark.resize(c + 1, 0); mark[c] = 1; }
            }
            int c = 0;
            while (c < (int)mark.size() && mark[c]) ++c;
            color[i] = c;
            P.colors = std::max(P.colors, c + 1);
        }
        for (int c = 0; c < P.colors; ++c) {
            P.cstart.push_back((int)P.brow0.size());
            for (int i = 0; i < n; ++i)
                if (color[i] == c && g.type[i] != 3) {
                    P.brow0.push_back(next);
                    idxVa[i] = next++;
                    if (g.type[i] == 1) idxVm[i] = next++;
                    P.bcnt.push_back(next - P.brow0.back());
                }
        }
        P.cstart.push_back((int)P.brow0.size());
    }
    P.rowbus.resize(P.nx); P.rowq.resize(P.nx); P.xbus.resize(P.nx); P.xkind.resize(P.nx);
    for (int i = 0; i < n; ++i) {       // the row of unknown Va_i is dP_i, the row of Vm_i is dQ_i
        if (idxVa[i] >= 0) { P.rowbus[idxVa[i]] = i; P.rowq[idxVa[i]] = 0; P.xbus[idxVa[i]] = i; P.xkind[idxVa[i]] = 0; }
        if (idxVm[i] >= 0) { P.rowbus[idxVm[i]] = i; P.rowq[idxVm[i]] = 1; P.xbus[idxVm[i]] = i; P.xkind[idxVm[i]] = 1; }
    }
    // --- Jacobian pattern: for every Ybus non-zero (i,j) up to four Jacobian
    // entries (dP/dVa, dP/dVm, dQ/dVa, dQ/dVm). Columns sorted within each row.
    P.J.n = P.nx;
    P.J.rowptr.assign(P.nx + 1, 0);
    struct E { int col, y, kind; };
    std::vector<E> row;
    for (int r = 0; r < P.nx; ++r) {
        const int i = P.rowbus[r];
        const int q = P.rowq[r];
        row.clear();
        for (int k = P.Y.rowptr[i]; k < P.Y.rowptr[i + 1]; ++k) {
            const int j = P.Y.col[k];
            if (idxVa[j] >= 0) row.push_back({idxVa[j], k, q ? 2 : 0});
            if (idxVm[j] >= 0) row.push_back({idxVm[j], k, q ? 3 : 1});
        }
        std::sort(row.begin(), row.end(), [](const E& a, const E& b) { return a.col < b.col; });
        for (const E& e : row) { P.J.col.push_back(e.col); P.jy.push_back(e.y); P.jkind.push_back(e.kind); P.jrow.push_back(i); }
        P.J.rowptr[r + 1] = (int)P.J.col.size();
    }
    P.J.nnz = (int)P.J.col.size();
    P.jdiag.assign(P.nx, -1);
    for (int r = 0; r < P.nx; ++r)
        for (int k = P.J.rowptr[r]; k < P.J.rowptr[r + 1]; ++k)
            if (P.J.col[k] == r) P.jdiag[r] = k;
    for (int r = 0; r < P.nx; ++r) if (P.jdiag[r] < 0) die("Jacobian row without diagonal entry");
    // --- dependency levels of the lower-triangular solve (how many sequential steps)
    std::vector<int> lvl(P.nx, 0);
    P.levels = 0;
    for (int r = 0; r < P.nx; ++r) {
        int l = 0;
        for (int k = P.J.rowptr[r]; k < P.J.rowptr[r + 1] && P.J.col[k] < r; ++k) l = std::max(l, lvl[P.J.col[k]] + 1);
        lvl[r] = l;
        P.levels = std::max(P.levels, l + 1);
    }
    return P;
}

// ---------------------------------------------------------------------------
//  Results and statistics shared by both solvers
// ---------------------------------------------------------------------------
struct PhaseTimes {                      // milliseconds, summed over Newton iterations
    double inject = 0, jacobian = 0, ilu = 0, solve = 0, update = 0, transfer = 0, setup = 0;
    double total() const { return inject + jacobian + ilu + solve + update; }
};

struct Solution {
    std::vector<double> Vm, Va;          // Va in radians
    std::vector<double> history;         // max |F| per Newton iteration
    std::vector<int> linear_iters;       // BiCGSTAB iterations per Newton step
    int newton_iters = 0;
    bool converged = false;
    PhaseTimes t;
    double wall_ms = 0;
};

static const double NR_TOL = 1e-8;        // pu, max power mismatch
static const int NR_MAX = 20;
static const int LIN_MAX = 10000;         // BiCGSTAB iteration cap
// Inexact Newton: the linear system only needs to be solved as accurately as the
// current mismatch warrants. Tolerance = 0.1 x max|F| (capped at 1e-2) keeps the
// quadratic convergence of Newton-Raphson while saving many early iterations.
static double forcing(double fmax) { return std::max(1e-12, std::min(1e-2, 0.1 * fmax)); }

// Independent power-balance check: rebuild I = Ybus V from scratch and return max |F|
static double max_mismatch(const Problem& P, const std::vector<double>& Vm, const std::vector<double>& Va) {
    std::vector<cplx> V(P.n), I(P.n);
    for (int i = 0; i < P.n; ++i) V[i] = mk(Vm[i] * std::cos(Va[i]), Vm[i] * std::sin(Va[i]));
    for (int i = 0; i < P.n; ++i) {
        cplx s = mk(0, 0);
        for (int k = P.Y.rowptr[i]; k < P.Y.rowptr[i + 1]; ++k) s = cadd(s, cmul(P.Yval[k], V[P.Y.col[k]]));
        I[i] = s;
    }
    double m = 0;
    for (int r = 0; r < P.nx; ++r)
        m = std::max(m, std::fabs(mismatch_entry(P.rowq[r], P.rowbus[r], V.data(), I.data(), P.Psp.data(),
                                                 P.Qsp.data())));
    return m;
}

// =============================================================================
//  CPU baseline: identical algorithm (Newton-Raphson + ILU(0)-preconditioned
//  BiCGSTAB). OpenMP parallelises the naturally parallel loops; the ILU(0)
//  factorisation and the triangular solves are sequential, as in standard
//  CPU sparse codes.
// =============================================================================
namespace cpu {

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void spmv(const Csr& A, const std::vector<double>& v, const double* x, double* y) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < A.n; ++i) {
        double s = 0;
        for (int k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) s += v[k] * x[A.col[k]];
        y[i] = s;
    }
}
static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    const int n = (int)a.size();
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// ILU(0): factors stored in place (unit lower L below the diagonal, U on and above)
static void ilu0(const Csr& A, const std::vector<int>& diag, std::vector<double>& LU, std::vector<int>& iw) {
    for (int i = 0; i < A.n; ++i) {
        for (int k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) iw[A.col[k]] = k;
        for (int k = A.rowptr[i]; k < diag[i]; ++k) {
            const int c = A.col[k];
            LU[k] /= LU[diag[c]];
            const double m = LU[k];
            for (int q = diag[c] + 1; q < A.rowptr[c + 1]; ++q) {
                const int p = iw[A.col[q]];
                if (p >= 0) LU[p] -= m * LU[q];
            }
        }
        for (int k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) iw[A.col[k]] = -1;
    }
}
static void lu_solve(const Csr& A, const std::vector<int>& diag, const std::vector<double>& LU,
                     const std::vector<double>& b, std::vector<double>& x) {
    for (int i = 0; i < A.n; ++i) {                   // L y = b  (unit diagonal)
        double s = b[i];
        for (int k = A.rowptr[i]; k < diag[i]; ++k) s -= LU[k] * x[A.col[k]];
        x[i] = s;
    }
    for (int i = A.n - 1; i >= 0; --i) {              // U x = y
        double s = x[i];
        for (int k = diag[i] + 1; k < A.rowptr[i + 1]; ++k) s -= LU[k] * x[A.col[k]];
        x[i] = s / LU[diag[i]];
    }
}

// Preconditioned BiCGSTAB (van der Vorst 1992). Returns iterations used, or -1.
static int bicgstab(const Csr& A, const std::vector<double>& Av, const std::vector<int>& diag,
                    const std::vector<double>& LU, const std::vector<double>& b, std::vector<double>& x, double tol) {
    const int n = A.n;
    std::vector<double> r(b), rh(b), p(n, 0), v(n, 0), ph(n), s(n), sh(n), t(n);
    std::fill(x.begin(), x.end(), 0.0);
    const double bn = std::sqrt(dot(b, b));
    if (bn == 0) return 0;
    double rho = 1, alpha = 1, omega = 1;
    for (int it = 1; it <= LIN_MAX; ++it) {
        const double rho1 = dot(rh, r);
        if (rho1 == 0) return -1;
        const double beta = (rho1 / rho) * (alpha / omega);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * (p[i] - omega * v[i]);
        lu_solve(A, diag, LU, p, ph);
        spmv(A, Av, ph.data(), v.data());
        alpha = rho1 / dot(rh, v);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];
        lu_solve(A, diag, LU, s, sh);
        spmv(A, Av, sh.data(), t.data());
        omega = dot(t, s) / dot(t, t);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) { x[i] += alpha * ph[i] + omega * sh[i]; r[i] = s[i] - omega * t[i]; }
        rho = rho1;
        if (std::sqrt(dot(r, r)) / bn < tol) return it;
        if (omega == 0) return -1;
    }
    return -1;
}

static Solution solve(const Problem& P, int threads) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    Solution S;
    const double w0 = now_ms();
    std::vector<double> Vm(P.Vm0), Va(P.Va0), F(P.nx), b(P.nx), dx(P.nx), Jv(P.J.nnz), LU(P.J.nnz);
    std::vector<cplx> V(P.n), I(P.n);
    std::vector<int> diag(P.nx), iw(P.nx, -1);
    for (int r = 0; r < P.nx; ++r) {
        diag[r] = -1;
        for (int k = P.J.rowptr[r]; k < P.J.rowptr[r + 1]; ++k) if (P.J.col[k] == r) diag[r] = k;
        if (diag[r] < 0) die("Jacobian has no diagonal entry");
    }
    for (int it = 0; it <= NR_MAX; ++it) {
        double t0 = now_ms();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < P.n; ++i) V[i] = mk(Vm[i] * std::cos(Va[i]), Vm[i] * std::sin(Va[i]));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < P.n; ++i) {                                  // I = Ybus * V
            cplx s = mk(0, 0);
            for (int k = P.Y.rowptr[i]; k < P.Y.rowptr[i + 1]; ++k) s = cadd(s, cmul(P.Yval[k], V[P.Y.col[k]]));
            I[i] = s;
        }
        double norm = 0;
#pragma omp parallel for reduction(max : norm) schedule(static)
        for (int r = 0; r < P.nx; ++r) {
            F[r] = mismatch_entry(P.rowq[r], P.rowbus[r], V.data(), I.data(), P.Psp.data(), P.Qsp.data());
            b[r] = -F[r];
            norm = std::max(norm, std::fabs(F[r]));
        }
        S.t.inject += now_ms() - t0;
        S.history.push_back(norm);
        if (norm < NR_TOL) { S.converged = true; S.newton_iters = it; break; }
        if (it == NR_MAX) break;

        t0 = now_ms();
#pragma omp parallel for schedule(static)
        for (int k = 0; k < P.J.nnz; ++k) {
            const int yk = P.jy[k];
            Jv[k] = jacobian_entry(P.jkind[k], P.jrow[k], P.Y.col[yk], P.Yval[yk], V.data(), Vm.data(), I.data());
        }
        S.t.jacobian += now_ms() - t0;

        t0 = now_ms();
        LU = Jv;
        ilu0(P.J, diag, LU, iw);
        S.t.ilu += now_ms() - t0;

        t0 = now_ms();
        const int li = bicgstab(P.J, Jv, diag, LU, b, dx, forcing(norm));
        if (li < 0) die("CPU BiCGSTAB failed");
        S.linear_iters.push_back(li);
        S.t.solve += now_ms() - t0;

        t0 = now_ms();
#pragma omp parallel for schedule(static)
        for (int r = 0; r < P.nx; ++r) {
            if (P.xkind[r] == 0) Va[P.xbus[r]] += dx[r]; else Vm[P.xbus[r]] += dx[r];
        }
        S.t.update += now_ms() - t0;
    }
    S.Vm = Vm; S.Va = Va;
    S.wall_ms = now_ms() - w0;
    return S;
}

}  // namespace cpu
