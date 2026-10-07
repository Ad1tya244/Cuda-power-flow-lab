// =============================================================================
//  pf_cuda.cu  -  CUDA Case Study: Smart Energy Grids (Power Grid Load Flow)
//
//  A grid operator must re-solve the load-flow (power-flow) equations of the
//  whole network every few seconds to see whether any bus voltage is drifting
//  towards a blackout. The equations are solved with the Newton-Raphson method.
//  Each Newton step needs:
//      I = Ybus * V                    sparse matrix-vector product   (cuSPARSE SpMV)
//      F = V .* conj(I) - S_specified  one thread per unknown         (custom kernel)
//      J = dF/dx                       one thread per NON-ZERO        (custom kernel)
//      J * dx = -F                     ILU(0)-preconditioned BiCGSTAB (cuSPARSE + cuBLAS)
//  Ybus and J are SPARSE: a bus is wired to ~3 neighbours, not to all n buses,
//  so > 99.9% of the matrix entries are zero and no thread ever touches them.
//
//  Modes
//    ./pf_cuda demo              IEEE 14-bus system, narrated 5-step pipeline
//    ./pf_cuda verify            correctness suite (IEEE 14 + synthetic grids)
//    ./pf_cuda bench             performance experiments E1-E5 -> results/*.csv
//    ./pf_cuda solve <case>      solve any case file
//    ./pf_cuda info              GPU information
//  Options: --out <dir> (default results), --threads <n> (CPU threads, default all)
// =============================================================================
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <filesystem>

#include "powerflow.h"

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t e_ = (call);                                                                \
        if (e_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,          \
                    cudaGetErrorString(e_));                                                    \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)
#define CUSPARSE_CHECK(call)                                                                    \
    do {                                                                                        \
        cusparseStatus_t s_ = (call);                                                           \
        if (s_ != CUSPARSE_STATUS_SUCCESS) {                                                    \
            fprintf(stderr, "cuSPARSE error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,      \
                    cusparseGetErrorString(s_));                                                \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)
#define CUBLAS_CHECK(call)                                                                      \
    do {                                                                                        \
        cublasStatus_t s_ = (call);                                                             \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                      \
            fprintf(stderr, "cuBLAS error %s at %s:%d: %d\n", #call, __FILE__, __LINE__, (int)s_); \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)

static std::string g_outdir = "results";
static int g_threads = 0;
static bool g_natural = false;
static const int BLOCK = 256;
static int grid_for(int n) { return (n + BLOCK - 1) / BLOCK; }

// =============================================================================
//  KERNELS  (every kernel checks its index against the array size: the last
//  block usually contains threads with no work)
// =============================================================================
__global__ void k_voltage(int n, const double* __restrict__ Vm, const double* __restrict__ Va, cplx* V) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s, c;
    sincos(Va[i], &s, &c);
    V[i] = mk(Vm[i] * c, Vm[i] * s);
}

// one thread per unknown: F = S(V) - S_specified, and b = -F for the linear solve
__global__ void k_mismatch(int nx, const int* __restrict__ rowq, const int* __restrict__ rowbus, const cplx* __restrict__ V,
                           const cplx* __restrict__ I, const double* __restrict__ Psp,
                           const double* __restrict__ Qsp, double* F, double* b) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nx) return;
    const double f = mismatch_entry(rowq[r], rowbus[r], V, I, Psp, Qsp);
    F[r] = f;
    b[r] = -f;
}

// one thread per NON-ZERO of the Jacobian: zeros are never stored or visited
__global__ void k_jacobian(int nnz, const int* __restrict__ jy, const int* __restrict__ jkind,
                           const int* __restrict__ jrow, const int* __restrict__ Ycol, const cplx* __restrict__ Y,
                           const cplx* __restrict__ V, const double* __restrict__ Vm, const cplx* __restrict__ I,
                           double* Jv) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    const int yk = jy[k];
    Jv[k] = jacobian_entry(jkind[k], jrow[k], Ycol[yk], Y[yk], V, Vm, I);
}

__global__ void k_update(int nx, const int* __restrict__ xbus, const int* __restrict__ xkind,
                         const double* __restrict__ dx, double* Va, double* Vm) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nx) return;
    if (xkind[r] == 0) Va[xbus[r]] += dx[r]; else Vm[xbus[r]] += dx[r];
}

// ---- ILU(0) preconditioner, colour by colour -------------------------------
// With the multicolour ordering, buses of one colour are never wired to each
// other, so all of them can be solved at the same time: ONE THREAD PER BUS of
// the current colour, touching only that bus's non-zeros. 4 colours -> the whole
// forward (L) solve is 4 short kernels instead of thousands of sequential rows.
__global__ void k_lsolve_colour(int kbeg, int kend, const int* __restrict__ brow0, const int* __restrict__ bcnt,
                                const int* __restrict__ rp, const int* __restrict__ col, const int* __restrict__ diag,
                                const double* __restrict__ LU, const double* __restrict__ b, double* y) {
    const int k = kbeg + blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= kend) return;
    const int r0 = brow0[k], r1 = r0 + bcnt[k];
    for (int r = r0; r < r1; ++r) {                       // L has a unit diagonal
        double s = b[r];
        for (int q = rp[r]; q < diag[r]; ++q) s -= LU[q] * y[col[q]];
        y[r] = s;
    }
}
__global__ void k_usolve_colour(int kbeg, int kend, const int* __restrict__ brow0, const int* __restrict__ bcnt,
                                const int* __restrict__ rp, const int* __restrict__ col, const int* __restrict__ diag,
                                const double* __restrict__ LU, const double* __restrict__ y, double* x) {
    const int k = kbeg + blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= kend) return;
    const int r0 = brow0[k], r1 = r0 + bcnt[k];
    for (int r = r1 - 1; r >= r0; --r) {
        double s = y[r];
        for (int q = diag[r] + 1; q < rp[r + 1]; ++q) s -= LU[q] * x[col[q]];
        x[r] = s / LU[diag[r]];
    }
}

// ---- BiCGSTAB: scalars live in device memory, so the host waits only once
// per iteration (to read the residual norm) instead of after every dot product.
enum { S_RHO, S_ALPHA, S_OMEGA, S_RHO1, S_RV, S_TS, S_TT, S_RN, S_BETA, S_COUNT };
__global__ void k_beta(double* sc) { sc[S_BETA] = (sc[S_RHO1] / sc[S_RHO]) * (sc[S_ALPHA] / sc[S_OMEGA]); }
__global__ void k_alpha(double* sc) { sc[S_ALPHA] = sc[S_RHO1] / sc[S_RV]; }
__global__ void k_omega(double* sc) { sc[S_OMEGA] = sc[S_TS] / sc[S_TT]; sc[S_RHO] = sc[S_RHO1]; }
__global__ void k_p_update(int n, double* p, const double* r, const double* v, const double* sc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = r[i] + sc[S_BETA] * (p[i] - sc[S_OMEGA] * v[i]);
}
__global__ void k_s_update(int n, double* s, const double* r, const double* v, const double* sc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) s[i] = r[i] - sc[S_ALPHA] * v[i];
}
__global__ void k_xr_update(int n, double* x, double* r, const double* s, const double* t, const double* ph,
                            const double* sh, const double* sc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += sc[S_ALPHA] * ph[i] + sc[S_OMEGA] * sh[i];
        r[i] = s[i] - sc[S_OMEGA] * t[i];
    }
}

// =============================================================================
//  GPU Newton-Raphson solver
// =============================================================================
struct GpuOptions {
    bool precondition = true;     // ILU(0) preconditioner on/off (experiment E5)
    bool colour_kernels = true;   // our colour-by-colour kernels (true) or cuSPARSE SpSV (false)
    bool narrate = false;
};

template <class T> static T* dalloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    return p;
}
template <class T> static void h2d(T* d, const std::vector<T>& h) {
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

struct EventTimer {
    cudaEvent_t a, b;
    EventTimer() { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~EventTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { CUDA_CHECK(cudaEventRecord(a)); }
    double stop() {
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};

static Solution gpu_solve(const Problem& P, const GpuOptions& opt = GpuOptions()) {
    Solution S;
    EventTimer tm;
    const double w0 = cpu::now_ms();
    const int n = P.n, nx = P.nx, ynnz = P.Y.nnz, jnnz = P.J.nnz;

    // ---------------- STEP 1: allocate device memory -------------------------
    tm.start();
    int *dYrow = dalloc<int>(n + 1), *dYcol = dalloc<int>(ynnz);
    cplx* dY = dalloc<cplx>(ynnz);
    int *dJrp = dalloc<int>(nx + 1), *dJcol = dalloc<int>(jnnz), *dJy = dalloc<int>(jnnz);
    int *dJkind = dalloc<int>(jnnz), *dJrow = dalloc<int>(jnnz);
    int *dRowbus = dalloc<int>(nx), *dRowq = dalloc<int>(nx), *dXbus = dalloc<int>(nx), *dXkind = dalloc<int>(nx);
    double *dJv = dalloc<double>(jnnz), *dLU = dalloc<double>(jnnz);
    double *dPsp = dalloc<double>(n), *dQsp = dalloc<double>(n), *dVm = dalloc<double>(n), *dVa = dalloc<double>(n);
    cplx *dV = dalloc<cplx>(n), *dI = dalloc<cplx>(n);
    double *dF = dalloc<double>(nx), *db = dalloc<double>(nx), *dx = dalloc<double>(nx);
    double *r = dalloc<double>(nx), *rh = dalloc<double>(nx), *p = dalloc<double>(nx), *v = dalloc<double>(nx);
    double *ph = dalloc<double>(nx), *s = dalloc<double>(nx), *sh = dalloc<double>(nx), *t = dalloc<double>(nx);
    double* ptmp = dalloc<double>(nx);                      // intermediate of the L-then-U solve
    double* dsc = dalloc<double>(S_COUNT);                  // BiCGSTAB scalars (device side)
    const bool colour = opt.colour_kernels && !P.cstart.empty();
    int *dB0 = dalloc<int>(P.brow0.size()), *dBc = dalloc<int>(P.bcnt.size()), *dDiag = dalloc<int>(nx);
    if (opt.narrate)
        printf("  [Step 1] cudaMalloc: Ybus (%d non-zeros), Jacobian (%d non-zeros), %d-unknown vectors\n",
               ynnz, jnnz, nx);

    // ---------------- STEP 2: copy the grid Host -> Device -------------------
    h2d(dYrow, P.Y.rowptr); h2d(dYcol, P.Y.col); h2d(dY, P.Yval);
    h2d(dJrp, P.J.rowptr); h2d(dJcol, P.J.col); h2d(dJy, P.jy); h2d(dJkind, P.jkind); h2d(dJrow, P.jrow);
    h2d(dRowbus, P.rowbus); h2d(dRowq, P.rowq); h2d(dXbus, P.xbus); h2d(dXkind, P.xkind);
    h2d(dPsp, P.Psp); h2d(dQsp, P.Qsp); h2d(dVm, P.Vm0); h2d(dVa, P.Va0);
    h2d(dDiag, P.jdiag);
    if (!P.brow0.empty()) { h2d(dB0, P.brow0); h2d(dBc, P.bcnt); }
    if (opt.narrate) printf("  [Step 2] cudaMemcpy H->D: Ybus, Jacobian pattern, specified powers, start voltages\n");

    // library handles and descriptors (created once, reused every Newton step)
    cusparseHandle_t sp;
    cublasHandle_t bl;
    CUSPARSE_CHECK(cusparseCreate(&sp));
    CUBLAS_CHECK(cublasCreate(&bl));
    cusparseSpMatDescr_t matY, matJ, matL, matU;
    CUSPARSE_CHECK(cusparseCreateCsr(&matY, n, n, ynnz, dYrow, dYcol, dY, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(&matJ, nx, nx, jnnz, dJrp, dJcol, dJv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(&matL, nx, nx, jnnz, dJrp, dJcol, dLU, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(&matU, nx, nx, jnnz, dJrp, dJcol, dLU, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseFillMode_t lower = CUSPARSE_FILL_MODE_LOWER, upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t unit = CUSPARSE_DIAG_TYPE_UNIT, nonunit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &lower, sizeof(lower)));
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &unit, sizeof(unit)));
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_FILL_MODE, &upper, sizeof(upper)));
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_DIAG_TYPE, &nonunit, sizeof(nonunit)));

    cusparseDnVecDescr_t vV, vI, vP, vPh, vV2, vS, vSh, vT, vTmp;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vV, n, dV, CUDA_C_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vI, n, dI, CUDA_C_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vP, nx, p, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vPh, nx, ph, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vV2, nx, v, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vS, nx, s, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vSh, nx, sh, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vT, nx, t, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vTmp, nx, ptmp, CUDA_R_64F));

    const cuDoubleComplex c1 = make_cuDoubleComplex(1, 0), c0 = make_cuDoubleComplex(0, 0);
    const double one = 1.0, zero = 0.0;
    size_t bY = 0, bJ = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &c1, matY, vV, &c0, vI, CUDA_C_64F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &bY));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matJ, vPh, &zero, vV2,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bJ));
    // Triangular-solve plans, one per (factor, vector pair): p -> ph and s -> sh
    struct Tri { cusparseSpSVDescr_t d; cusparseSpMatDescr_t m; cusparseDnVecDescr_t x, y; void* buf; };
    Tri tri[4] = {{nullptr, matL, vP, vTmp, nullptr}, {nullptr, matU, vTmp, vPh, nullptr},
                  {nullptr, matL, vS, vTmp, nullptr}, {nullptr, matU, vTmp, vSh, nullptr}};
    for (Tri& q : tri) {
        size_t bs = 0;
        CUSPARSE_CHECK(cusparseSpSV_createDescr(&q.d));
        CUSPARSE_CHECK(cusparseSpSV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, q.m, q.x, q.y, CUDA_R_64F,
                                               CUSPARSE_SPSV_ALG_DEFAULT, q.d, &bs));
        q.buf = dalloc<char>(bs);
    }
    void *bufY = dalloc<char>(bY), *bufJ = dalloc<char>(bJ);

    // ILU(0): structural analysis once (the Jacobian pattern never changes)
    cusparseMatDescr_t descrM;
    csrilu02Info_t ilu;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrM));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrM, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseCreateCsrilu02Info(&ilu));
    int bIlu = 0;
    CUSPARSE_CHECK(cusparseDcsrilu02_bufferSize(sp, nx, jnnz, descrM, dLU, dJrp, dJcol, ilu, &bIlu));
    void* bufIlu = dalloc<char>((size_t)bIlu);
    CUSPARSE_CHECK(cusparseDcsrilu02_analysis(sp, nx, jnnz, descrM, dLU, dJrp, dJcol, ilu,
                                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, bufIlu));
    S.t.setup = tm.stop();

    // Preconditioner M^{-1} = (LU)^{-1}: which = 0 for p -> ph, 1 for s -> sh
    auto precond = [&](int which) {
        const double* in = which ? s : p;
        double* out = which ? sh : ph;
        if (!opt.precondition) { CUDA_CHECK(cudaMemcpy(out, in, nx * sizeof(double), cudaMemcpyDeviceToDevice)); return; }
        if (colour) {
            for (int c = 0; c < P.colors; ++c) {                     // forward: colour 0, 1, 2, ...
                const int kb = P.cstart[c], ke = P.cstart[c + 1];
                if (ke > kb) k_lsolve_colour<<<grid_for(ke - kb), BLOCK>>>(kb, ke, dB0, dBc, dJrp, dJcol, dDiag, dLU, in, ptmp);
            }
            for (int c = P.colors - 1; c >= 0; --c) {                // backward: ..., 2, 1, 0
                const int kb = P.cstart[c], ke = P.cstart[c + 1];
                if (ke > kb) k_usolve_colour<<<grid_for(ke - kb), BLOCK>>>(kb, ke, dB0, dBc, dJrp, dJcol, dDiag, dLU, ptmp, out);
            }
            return;
        }
        for (int k = 2 * which; k < 2 * which + 2; ++k)
            CUSPARSE_CHECK(cusparseSpSV_solve(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, tri[k].m, tri[k].x, tri[k].y,
                                              CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, tri[k].d));
    };
    auto spmvJ = [&](cusparseDnVecDescr_t in, cusparseDnVecDescr_t out) {
        CUSPARSE_CHECK(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matJ, in, &zero, out, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, bufJ));
    };
    auto dnrm = [&](const double* a) {
        double res;
        CUBLAS_CHECK(cublasDnrm2(bl, nx, a, 1, &res));
        return res;
    };

    // ---------------- STEP 3: Newton-Raphson iterations (kernels + libraries) --
    if (opt.narrate)
        printf("  [Step 3] Newton loop: k_voltage<<<%d,%d>>>, cusparseSpMV (Ybus*V), k_mismatch<<<%d,%d>>>,\n"
               "           k_jacobian<<<%d,%d>>> (one thread per non-zero), ILU(0) + BiCGSTAB (cuSPARSE/cuBLAS)\n",
               grid_for(n), BLOCK, grid_for(nx), BLOCK, grid_for(jnnz), BLOCK);
    for (int it = 0; it <= NR_MAX; ++it) {
        tm.start();
        k_voltage<<<grid_for(n), BLOCK>>>(n, dVm, dVa, dV);
        CUDA_CHECK(cudaGetLastError());
        CUSPARSE_CHECK(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &c1, matY, vV, &c0, vI, CUDA_C_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, bufY));
        k_mismatch<<<grid_for(nx), BLOCK>>>(nx, dRowq, dRowbus, dV, dI, dPsp, dQsp, dF, db);
        CUDA_CHECK(cudaGetLastError());
        int imax = 1;
        CUBLAS_CHECK(cublasIdamax(bl, nx, dF, 1, &imax));
        double fmax = 0;
        CUDA_CHECK(cudaMemcpy(&fmax, dF + (imax - 1), sizeof(double), cudaMemcpyDeviceToHost));
        fmax = std::fabs(fmax);
        S.t.inject += tm.stop();
        S.history.push_back(fmax);
        if (opt.narrate) printf("           iteration %d: max mismatch = %.3e pu\n", it, fmax);
        if (fmax < NR_TOL) { S.converged = true; S.newton_iters = it; break; }
        if (it == NR_MAX) break;

        tm.start();
        k_jacobian<<<grid_for(jnnz), BLOCK>>>(jnnz, dJy, dJkind, dJrow, dYcol, dY, dV, dVm, dI, dJv);
        CUDA_CHECK(cudaGetLastError());
        S.t.jacobian += tm.stop();

        tm.start();
        if (opt.precondition) {
            CUDA_CHECK(cudaMemcpy(dLU, dJv, jnnz * sizeof(double), cudaMemcpyDeviceToDevice));
            CUSPARSE_CHECK(cusparseDcsrilu02(sp, nx, jnnz, descrM, dLU, dJrp, dJcol, ilu,
                                             CUSPARSE_SOLVE_POLICY_USE_LEVEL, bufIlu));
            int zp = -1;
            if (cusparseXcsrilu02_zeroPivot(sp, ilu, &zp) == CUSPARSE_STATUS_ZERO_PIVOT) die("ILU(0) zero pivot");
            if (!colour)                             // new factor values -> redo the cuSPARSE solve analysis
                for (Tri& q : tri)
                    CUSPARSE_CHECK(cusparseSpSV_analysis(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, q.m, q.x, q.y,
                                                         CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, q.d, q.buf));
        }
        S.t.ilu += tm.stop();

        // ---- BiCGSTAB: J dx = b (= -F), starting from dx = 0
        tm.start();
        CUDA_CHECK(cudaMemset(dx, 0, nx * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(r, db, nx * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(rh, db, nx * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(p, 0, nx * sizeof(double)));
        CUDA_CHECK(cudaMemset(v, 0, nx * sizeof(double)));
        const double bn = dnrm(db), lin_tol = forcing(fmax);
        const double init[S_COUNT] = {1, 1, 1, 0, 0, 0, 0, 0, 0};          // rho = alpha = omega = 1
        CUDA_CHECK(cudaMemcpy(dsc, init, sizeof init, cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSetPointerMode(bl, CUBLAS_POINTER_MODE_DEVICE));   // dot results stay on the GPU
        int used = -1;
        double sc[S_COUNT];
        for (int li = 1; li <= LIN_MAX && bn > 0; ++li) {
            const int g = grid_for(nx);
            CUBLAS_CHECK(cublasDdot(bl, nx, rh, 1, r, 1, dsc + S_RHO1));      // rho1 = (rh, r)
            k_beta<<<1, 1>>>(dsc);
            k_p_update<<<g, BLOCK>>>(nx, p, r, v, dsc);                       // p = r + beta (p - omega v)
            precond(0);                                                       // ph = M^-1 p
            spmvJ(vPh, vV2);                                                  // v  = J ph
            CUBLAS_CHECK(cublasDdot(bl, nx, rh, 1, v, 1, dsc + S_RV));
            k_alpha<<<1, 1>>>(dsc);                                           // alpha = rho1 / (rh, v)
            k_s_update<<<g, BLOCK>>>(nx, s, r, v, dsc);                       // s = r - alpha v
            precond(1);                                                       // sh = M^-1 s
            spmvJ(vSh, vT);                                                   // t  = J sh
            CUBLAS_CHECK(cublasDdot(bl, nx, t, 1, s, 1, dsc + S_TS));
            CUBLAS_CHECK(cublasDdot(bl, nx, t, 1, t, 1, dsc + S_TT));
            k_omega<<<1, 1>>>(dsc);                                           // omega = (t, s) / (t, t)
            k_xr_update<<<g, BLOCK>>>(nx, dx, r, s, t, ph, sh, dsc);          // x += ..., r = s - omega t
            CUBLAS_CHECK(cublasDnrm2(bl, nx, r, 1, dsc + S_RN));
            CUDA_CHECK(cudaMemcpy(sc, dsc, sizeof sc, cudaMemcpyDeviceToHost)); // the only wait per iteration
            if (!std::isfinite(sc[S_RN]) || sc[S_RHO1] == 0 || sc[S_OMEGA] == 0) break;   // breakdown
            if (sc[S_RN] / bn < lin_tol) { used = li; break; }
        }
        CUBLAS_CHECK(cublasSetPointerMode(bl, CUBLAS_POINTER_MODE_HOST));
        CUDA_CHECK(cudaGetLastError());
        S.t.solve += tm.stop();
        S.linear_iters.push_back(used);
        if (used < 0) {                                   // report, do not abort: E5 studies this case
            S.converged = false;
            S.newton_iters = it;
            break;
        }

        tm.start();
        k_update<<<grid_for(nx), BLOCK>>>(nx, dXbus, dXkind, dx, dVa, dVm);
        CUDA_CHECK(cudaGetLastError());
        S.t.update += tm.stop();
    }

    // ---------------- STEP 4: copy the solution Device -> Host ----------------
    tm.start();
    S.Vm.resize(n); S.Va.resize(n);
    CUDA_CHECK(cudaMemcpy(S.Vm.data(), dVm, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(S.Va.data(), dVa, n * sizeof(double), cudaMemcpyDeviceToHost));
    S.t.transfer = tm.stop();
    if (opt.narrate) printf("  [Step 4] cudaMemcpy D->H: %d voltage magnitudes and angles\n", n);

    // ---------------- STEP 5: free everything ----------------------------------
    cusparseDestroyCsrilu02Info(ilu); cusparseDestroyMatDescr(descrM);
    for (Tri& q : tri) { cusparseSpSV_destroyDescr(q.d); CUDA_CHECK(cudaFree(q.buf)); }
    for (auto m : {matY, matJ, matL, matU}) cusparseDestroySpMat(m);
    for (auto d : {vV, vI, vP, vPh, vV2, vS, vSh, vT, vTmp}) cusparseDestroyDnVec(d);
    cusparseDestroy(sp); cublasDestroy(bl);
    for (void* q : {(void*)dYrow, (void*)dYcol, (void*)dY, (void*)dJrp, (void*)dJcol, (void*)dJy, (void*)dJkind,
                    (void*)dJrow, (void*)dRowbus, (void*)dRowq, (void*)dXbus, (void*)dXkind, (void*)dJv, (void*)dLU,
                    (void*)dPsp, (void*)dQsp, (void*)dVm, (void*)dVa, (void*)dV, (void*)dI, (void*)dF, (void*)db,
                    (void*)dx, (void*)r, (void*)rh, (void*)p, (void*)v, (void*)ph, (void*)s, (void*)sh, (void*)t,
                    (void*)ptmp, (void*)dsc, (void*)dB0, (void*)dBc, (void*)dDiag, bufY, bufJ, bufIlu})
        CUDA_CHECK(cudaFree(q));
    if (opt.narrate) printf("  [Step 5] cudaFree: all device buffers and library handles released\n");
    S.wall_ms = cpu::now_ms() - w0;
    return S;
}

// =============================================================================
//  HELPERS
// =============================================================================
static int all_threads() {
#ifdef _OPENMP
    return omp_get_num_procs();
#else
    return 1;
#endif
}
static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
static const double RAD2DEG = 57.29577951308232;
static FILE* open_out(const std::string& name, const char* header = nullptr) {
    std::filesystem::create_directories(g_outdir);
    FILE* f = fopen((g_outdir + "/" + name).c_str(), "w");
    if (!f) die(("cannot write " + g_outdir + "/" + name).c_str());
    if (header) fprintf(f, "%s\n", header);
    return f;
}
static void write_solution(const Solution& S, const std::string& name) {
    FILE* f = open_out(name);
    for (size_t i = 0; i < S.Vm.size(); ++i) fprintf(f, "%zu %.15f %.15f\n", i + 1, S.Vm[i], S.Va[i] * RAD2DEG);
    fclose(f);
}
static int sum(const std::vector<int>& v) { int s = 0; for (int x : v) s += x; return s; }

static void print_problem(const Grid& g, const Problem& P) {
    const double dens = 100.0 * P.Y.nnz / ((double)P.n * P.n);
    int npv = 0;
    for (int ty : g.type) npv += (ty != 1);
    printf("  grid: %d buses (%d generator buses incl. slack, %d load buses), %lld lines | Ybus %d non-zeros (%.4f%% of %d x %d) | "
           "Jacobian %d x %d, %d non-zeros, %d bus colours -> %d triangular-solve levels\n",
           P.n, npv, P.npq, P.branches, P.Y.nnz, dens, P.n, P.n, P.nx, P.nx, P.J.nnz, P.colors, P.levels);
}

// =============================================================================
//  MODE: info
// =============================================================================
static void print_info(FILE* f) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int rt = 0, drv = 0, csp = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    cusparseHandle_t h;
    CUSPARSE_CHECK(cusparseCreate(&h));
    CUSPARSE_CHECK(cusparseGetVersion(h, &csp));
    cusparseDestroy(h);
    fprintf(f, "GPU     : %s, compute capability %d.%d, %d SMs, %d MHz, %.0f MiB, L2 %d KiB, "
               "memory bus %d-bit\n", p.name, p.major, p.minor, p.multiProcessorCount, p.clockRate / 1000,
            p.totalGlobalMem / 1048576.0, p.l2CacheSize / 1024, p.memoryBusWidth);
    fprintf(f, "CUDA    : runtime %d.%d, driver %d.%d, cuSPARSE %d\n", rt / 1000, (rt % 1000) / 10, drv / 1000,
            (drv % 1000) / 10, csp);
    fprintf(f, "CPU     : %d hardware threads (OpenMP)\n", all_threads());
}

// =============================================================================
//  MODE: demo  - IEEE 14-bus system, narrated
// =============================================================================
static int run_demo() {
    printf("\n================ DEMO: IEEE 14-bus test system on the GPU ================\n");
    Grid g = read_case("cases/ieee14.case");
    Problem P = build_problem(g);
    print_problem(g, P);
    printf("\nGPU Newton-Raphson load flow (the 5 CUDA steps):\n");
    GpuOptions o;
    o.narrate = true;
    Solution G = gpu_solve(P, o);
    Solution C = cpu::solve(P, 1);
    printf("\n%-4s %-6s | %-18s | %-18s | %s\n", "bus", "type", "GPU  Vm (pu)  Va", "published Vm  Va", "CPU = GPU?");
    printf("-----------+--------------------+--------------------+-----------\n");
    const char* tn[] = {"", "load", "gen", "slack"};
    for (int i = 0; i < P.n; ++i)
        printf("%-4d %-6s | %7.4f %9.3f  | %6.3f %9.2f   | %s\n", i + 1, tn[g.type[i]], G.Vm[i], G.Va[i] * RAD2DEG,
               g.VmPub[i], g.VaPub[i], (std::fabs(G.Vm[i] - C.Vm[i]) < 1e-9 && std::fabs(G.Va[i] - C.Va[i]) < 1e-9)
                                           ? "yes" : "NO");
    const double dvm = max_abs_diff(G.Vm, g.VmPub);
    std::vector<double> vad(P.n);
    for (int i = 0; i < P.n; ++i) vad[i] = G.Va[i] * RAD2DEG;
    const double dva = max_abs_diff(vad, g.VaPub);
    const double mm = max_mismatch(P, G.Vm, G.Va);
    printf("\nConverged in %d Newton iterations; power balance satisfied to %.1e pu at every bus.\n",
           G.newton_iters, mm);
    printf("Largest difference from the published solution: %.4f pu, %.3f deg (table rounded to 3/2 decimals)\n",
           dvm, dva);
    const bool ok = G.converged && mm < NR_TOL && dvm < 0.002 && dva < 0.02;
    printf("DEMO RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// =============================================================================
//  MODE: verify
// =============================================================================
static bool check(const char* what, bool ok, const char* detail) {
    printf("  %-62s : %s  %s\n", what, ok ? "PASS" : "FAIL", detail);
    return ok;
}
static int run_verify() {
    printf("\n================ VERIFY ================\n");
    bool all = true;
    char d[160];
    {
        Grid g = read_case("cases/ieee14.case");
        Problem P = build_problem(g), Pn = build_problem(g, Ordering::NATURAL);
        Solution G = gpu_solve(P), C = cpu::solve(Pn, 1);
        std::vector<double> vad(P.n);
        for (int i = 0; i < P.n; ++i) vad[i] = G.Va[i] * RAD2DEG;
        const double dvm = max_abs_diff(G.Vm, g.VmPub), dva = max_abs_diff(vad, g.VaPub);
        snprintf(d, sizeof d, "(max diff %.4f pu, %.3f deg)", dvm, dva);
        all &= check("IEEE 14-bus: GPU matches published solution", dvm < 0.002 && dva < 0.02, d);
        snprintf(d, sizeof d, "(max diff %.1e pu, %.1e rad)", max_abs_diff(G.Vm, C.Vm), max_abs_diff(G.Va, C.Va));
        all &= check("IEEE 14-bus: GPU equals CPU", max_abs_diff(G.Vm, C.Vm) < 1e-9 && max_abs_diff(G.Va, C.Va) < 1e-9, d);
        snprintf(d, sizeof d, "(%.1e pu)", max_mismatch(P, G.Vm, G.Va));
        all &= check("IEEE 14-bus: power balance < 1e-8 pu at every bus", max_mismatch(P, G.Vm, G.Va) < NR_TOL, d);
    }
    for (int n : {1000, 10000, 100000}) {
        Grid g = synthetic_grid(n);
        Problem P = build_problem(g), Pn = build_problem(g, Ordering::NATURAL);
        Solution G = gpu_solve(P), C = cpu::solve(Pn, all_threads());
        const double mmG = max_mismatch(P, G.Vm, G.Va), mmC = max_mismatch(P, C.Vm, C.Va);
        char what[96];
        snprintf(what, sizeof what, "%d-bus grid: GPU converged, power balance < 1e-8 pu", n);
        snprintf(d, sizeof d, "(%d Newton its, mismatch %.1e pu)", G.newton_iters, mmG);
        all &= check(what, G.converged && mmG < NR_TOL, d);
        snprintf(what, sizeof what, "%d-bus grid: CPU converged, power balance < 1e-8 pu", n);
        snprintf(d, sizeof d, "(%d Newton its, mismatch %.1e pu)", C.newton_iters, mmC);
        all &= check(what, C.converged && mmC < NR_TOL, d);
        const double dvm = max_abs_diff(G.Vm, C.Vm), dva = max_abs_diff(G.Va, C.Va);
        snprintf(what, sizeof what, "%d-bus grid: GPU and CPU voltages agree", n);
        snprintf(d, sizeof d, "(max diff %.1e pu, %.1e rad)", dvm, dva);
        all &= check(what, dvm < 1e-7 && dva < 1e-7, d);
        if (n == 10000 || n == 100000) {                    // for the independent Python check
            write_case(g, g_outdir + "/synthetic" + std::to_string(n) + ".case");
            write_solution(G, "synthetic" + std::to_string(n) + "_gpu.sol");
        }
    }
    printf("  wrote results/synthetic10000.case (+ _gpu.sol) and 100000 for tools/pf_reference.py --compare\n");
    printf("VERIFY RESULT: %s\n", all ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all ? 0 : 1;
}

// =============================================================================
//  MODE: bench
// =============================================================================
// E3: why sparsity matters - dense cuBLAS Zgemv vs sparse cuSPARSE SpMV for I = Ybus*V
static void bench_dense_vs_sparse() {
    printf("\n[E3] I = Ybus * V : dense (cuBLAS Zgemv) vs sparse (cuSPARSE SpMV), average of 50 runs\n");
    FILE* f = open_out("e3_dense_vs_sparse.csv", "buses,ybus_nnz,density_pct,dense_mb,sparse_mb,dense_ms,sparse_ms");
    cublasHandle_t bl;
    cusparseHandle_t sp;
    CUBLAS_CHECK(cublasCreate(&bl));
    CUSPARSE_CHECK(cusparseCreate(&sp));
    for (int n : {1000, 2000, 4000, 6000}) {
        Problem P = build_problem(synthetic_grid(n));
        std::vector<cplx> dense((size_t)n * n, mk(0, 0));                 // column-major for cuBLAS
        for (int i = 0; i < n; ++i)
            for (int k = P.Y.rowptr[i]; k < P.Y.rowptr[i + 1]; ++k) dense[(size_t)P.Y.col[k] * n + i] = P.Yval[k];
        std::vector<cplx> V(n);
        for (int i = 0; i < n; ++i) V[i] = mk(1.0, 0.01 * (i % 7));
        cplx *dA = dalloc<cplx>((size_t)n * n), *dV = dalloc<cplx>(n), *dI = dalloc<cplx>(n), *dY = dalloc<cplx>(P.Y.nnz);
        int *dr = dalloc<int>(n + 1), *dc = dalloc<int>(P.Y.nnz);
        h2d(dA, dense); h2d(dV, V); h2d(dY, P.Yval); h2d(dr, P.Y.rowptr); h2d(dc, P.Y.col);
        const cuDoubleComplex c1 = make_cuDoubleComplex(1, 0), c0 = make_cuDoubleComplex(0, 0);
        EventTimer tm;
        const int R = 50;
        CUBLAS_CHECK(cublasZgemv(bl, CUBLAS_OP_N, n, n, &c1, (cuDoubleComplex*)dA, n, (cuDoubleComplex*)dV, 1, &c0,
                                 (cuDoubleComplex*)dI, 1));
        tm.start();
        for (int k = 0; k < R; ++k)
            CUBLAS_CHECK(cublasZgemv(bl, CUBLAS_OP_N, n, n, &c1, (cuDoubleComplex*)dA, n, (cuDoubleComplex*)dV, 1,
                                     &c0, (cuDoubleComplex*)dI, 1));
        const double tDense = tm.stop() / R;
        std::vector<cplx> Id(n), Is(n);
        CUDA_CHECK(cudaMemcpy(Id.data(), dI, n * sizeof(cplx), cudaMemcpyDeviceToHost));
        cusparseSpMatDescr_t m;
        cusparseDnVecDescr_t vx, vy;
        CUSPARSE_CHECK(cusparseCreateCsr(&m, n, n, P.Y.nnz, dr, dc, dY, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vx, n, dV, CUDA_C_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vy, n, dI, CUDA_C_64F));
        size_t bs = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &c1, m, vx, &c0, vy, CUDA_C_64F,
                                               CUSPARSE_SPMV_ALG_DEFAULT, &bs));
        void* buf = dalloc<char>(bs);
        CUSPARSE_CHECK(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &c1, m, vx, &c0, vy, CUDA_C_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, buf));
        tm.start();
        for (int k = 0; k < R; ++k)
            CUSPARSE_CHECK(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &c1, m, vx, &c0, vy, CUDA_C_64F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, buf));
        const double tSparse = tm.stop() / R;
        CUDA_CHECK(cudaMemcpy(Is.data(), dI, n * sizeof(cplx), cudaMemcpyDeviceToHost));
        double diff = 0;
        for (int i = 0; i < n; ++i) diff = std::max(diff, std::fabs(Id[i].re - Is[i].re) + std::fabs(Id[i].im - Is[i].im));
        if (diff > 1e-8) die("dense and sparse products differ");
        const double dmb = (double)n * n * 16 / 1048576.0, smb = (P.Y.nnz * 20.0 + (n + 1) * 4.0) / 1048576.0;
        printf("  %5d buses: dense %8.3f ms (%7.1f MB) | sparse %7.4f ms (%5.2f MB) | %6.0fx faster, "
               "%.3f%% of entries non-zero\n", n, tDense, dmb, tSparse, smb, tDense / tSparse,
               100.0 * P.Y.nnz / ((double)n * n));
        fprintf(f, "%d,%d,%.5f,%.2f,%.4f,%.5f,%.6f\n", n, P.Y.nnz, 100.0 * P.Y.nnz / ((double)n * n), dmb, smb,
                tDense, tSparse);
        cusparseDestroySpMat(m); cusparseDestroyDnVec(vx); cusparseDestroyDnVec(vy);
        for (void* q : {(void*)dA, (void*)dV, (void*)dI, (void*)dY, (void*)dr, (void*)dc, buf}) CUDA_CHECK(cudaFree(q));
    }
    cublasDestroy(bl);
    cusparseDestroy(sp);
    fclose(f);
}

static int run_bench(const std::vector<int>& sizes) {
    printf("\n================ BENCHMARKS ================\n");
    print_info(stdout);
    { FILE* f = open_out("device_info.txt"); print_info(f); fclose(f); }
    const int T = all_threads();
    { Problem w = build_problem(synthetic_grid(1000)); gpu_solve(w); }        // warm-up (context, libraries)

    printf("\n[E1] Full Newton-Raphson load flow vs grid size (CPU: 1 and %d threads)\n", T);
    printf("%8s %9s %8s | %5s %6s | %10s %10s %10s | %7s %7s\n", "buses", "J nnz", "unknowns", "NR", "BiCG",
           "GPU ms", "CPU1 ms", "CPU" "N ms", "vs 1T", "vs NT");
    FILE* f1 = open_out("e1_scaling.csv",
                        "buses,lines,ybus_nnz,unknowns,jac_nnz,newton_its,bicg_its_gpu,gpu_loop_ms,gpu_setup_ms,"
                        "gpu_transfer_ms,gpu_wall_ms,cpu1_ms,cpuN_ms,cpu_threads,bicg_its_cpu1,max_mismatch_gpu");
    FILE* f2 = open_out("e2_phases.csv",
                        "buses,device,inject_ms,jacobian_ms,ilu_ms,solve_ms,update_ms");
    FILE* f4 = open_out("e4_convergence.csv", "buses,iteration,max_mismatch_pu,bicg_iterations");
    for (int n : sizes) {
        Grid g = synthetic_grid(n);
        Problem P = build_problem(g);                                  // GPU: multicolour ordering
        Problem Pn = build_problem(g, Ordering::NATURAL);              // CPU: natural ordering (its best)
        Solution G1 = gpu_solve(P), G2 = gpu_solve(P);
        const Solution& G = (G1.t.total() < G2.t.total()) ? G1 : G2;         // best of 2
        Solution C1 = cpu::solve(Pn, 1);
        Solution CN = cpu::solve(Pn, T);
        const double mm = max_mismatch(P, G.Vm, G.Va);
        if (!G.converged || !C1.converged || !CN.converged || mm >= NR_TOL) die("benchmark run did not converge");
        if (max_abs_diff(G.Vm, C1.Vm) > 1e-7 || max_abs_diff(G.Va, C1.Va) > 1e-7) die("GPU and CPU disagree");
        printf("%8d %9d %8d | %5d %6d | %10.1f %10.1f %10.1f | %6.1fx %6.1fx\n", n, P.J.nnz, P.nx, G.newton_iters,
               sum(G.linear_iters), G.t.total(), C1.t.total(), CN.t.total(), C1.t.total() / G.t.total(),
               CN.t.total() / G.t.total());
        fprintf(f1, "%d,%lld,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%.3e\n", n, P.branches, P.Y.nnz, P.nx,
                P.J.nnz, G.newton_iters, sum(G.linear_iters), G.t.total(), G.t.setup, G.t.transfer, G.wall_ms,
                C1.t.total(), CN.t.total(), T, sum(C1.linear_iters), mm);
        const char* names[3] = {"gpu", "cpu1", "cpuN"};
        const Solution* sols[3] = {&G, &C1, &CN};
        for (int q = 0; q < 3; ++q)
            fprintf(f2, "%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f\n", n, names[q], sols[q]->t.inject, sols[q]->t.jacobian,
                    sols[q]->t.ilu, sols[q]->t.solve, sols[q]->t.update);
        for (size_t k = 0; k < G.history.size(); ++k)
            fprintf(f4, "%d,%zu,%.3e,%d\n", n, k, G.history[k], k < G.linear_iters.size() ? G.linear_iters[k] : 0);
        fflush(f1); fflush(f2); fflush(f4);
    }
    fclose(f1); fclose(f2); fclose(f4);

    printf("\n[E2] Time per phase at %d buses (ms, summed over all Newton steps): see e2_phases.csv\n", sizes.back());

    bench_dense_vs_sparse();

    // E5: how the linear solver was made GPU-friendly (design variants on the same grid)
    printf("\n[E5] Linear-solver design variants (whole Newton-Raphson solve)\n");
    FILE* f5 = open_out("e5_solver_variants.csv",
                        "buses,variant,device,ordering,trisolve,levels,newton_its,bicg_its,solve_ms,total_ms,converged");
    for (int n : {10000, 50000}) {
        Grid g = synthetic_grid(n);
        Problem Pm = build_problem(g), Pn = build_problem(g, Ordering::NATURAL);
        struct V { const char* name; bool gpu; bool mc; bool colour; bool pre; };
        const V vs[] = {{"GPU natural order + cuSPARSE SpSV", true, false, false, true},
                        {"GPU multicolour + cuSPARSE SpSV", true, true, false, true},
                        {"GPU multicolour + colour kernels", true, true, true, true},
                        {"GPU multicolour, no preconditioner", true, true, true, false},
                        {"CPU 1 thread, natural order", false, false, false, true},
                        {"CPU 1 thread, multicolour", false, true, false, true}};
        for (const V& v : vs) {
            if (n > 10000 && v.gpu && (!v.mc || !v.pre)) continue;          // slow/failing variants: 10k only
            const Problem& P = v.mc ? Pm : Pn;
            Solution S;
            if (v.gpu) { GpuOptions o; o.colour_kernels = v.colour; o.precondition = v.pre; S = gpu_solve(P, o); }
            else S = cpu::solve(P, 1);
            int its = 0;
            for (int li : S.linear_iters) its += (li > 0 ? li : LIN_MAX);
            printf("  %6d buses | %-36s | levels %4d | Newton %2d | BiCGSTAB %5d | solve %9.1f ms | total %9.1f ms %s\n",
                   n, v.name, P.levels, S.newton_iters, its, S.t.solve, S.t.total(), S.converged ? "" : "(NOT converged)");
            fprintf(f5, "%d,%s,%s,%s,%s,%d,%d,%d,%.3f,%.3f,%d\n", n, v.name, v.gpu ? "gpu" : "cpu1",
                    v.mc ? "multicolour" : "natural", v.gpu ? (v.pre ? (v.colour ? "colour-kernels" : "cusparse-spsv") : "none")
                    : "sequential", P.levels, S.newton_iters, its, S.t.solve, S.t.total(), S.converged ? 1 : 0);
            fflush(f5);
        }
    }
    fclose(f5);
    printf("\nBENCH COMPLETE - CSV files in %s/\n", g_outdir.c_str());
    return 0;
}

// =============================================================================
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string mode = "all", casefile;
    std::vector<int> sizes = {1000, 5000, 10000, 20000, 50000, 100000, 200000};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) g_outdir = argv[++i];
        else if (a == "--threads" && i + 1 < argc) g_threads = atoi(argv[++i]);
        else if (a == "--natural") g_natural = true;
        else if (a == "--sizes" && i + 1 < argc) {
            sizes.clear();
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) sizes.push_back(atoi(tok.c_str()));
        } else if (a == "demo" || a == "verify" || a == "bench" || a == "info" || a == "all" || a == "probe" || a == "cpu") mode = a;
        else if (a == "solve" && i + 1 < argc) { mode = "solve"; casefile = argv[++i]; }
        else { fprintf(stderr, "usage: %s [demo|verify|bench|info|all|solve <case>] [--sizes a,b,..] [--out dir]\n", argv[0]); return 2; }
    }
    int dev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev));
    if (dev == 0) die("no CUDA device found");
#ifdef _OPENMP
    if (g_threads > 0) omp_set_num_threads(g_threads);
#endif
    if (mode == "info") { print_info(stdout); return 0; }
    if (mode == "cpu") {                           // CPU baseline only (natural ordering), --threads t
        const int T = g_threads > 0 ? g_threads : all_threads();
        FILE* f = open_out("e1_cpu_t" + std::to_string(T) + ".csv", "buses,threads,cpu_ms,inject_ms,jacobian_ms,ilu_ms,solve_ms,bicg_its");
        for (int n : sizes) {
            Problem Pn = build_problem(synthetic_grid(n), Ordering::NATURAL);
            Solution C = cpu::solve(Pn, T);
            printf("n=%d threads=%d total=%.1f ms (inject %.1f, jacobian %.1f, ilu %.1f, solve %.1f) bicg=%d converged=%d\n",
                   n, T, C.t.total(), C.t.inject, C.t.jacobian, C.t.ilu, C.t.solve, sum(C.linear_iters), (int)C.converged);
            fprintf(f, "%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n", n, T, C.t.total(), C.t.inject, C.t.jacobian, C.t.ilu,
                    C.t.solve, sum(C.linear_iters));
            fflush(f);
        }
        fclose(f);
        return 0;
    }
    if (mode == "probe") {                         // diagnostics: GPU only, synthetic grids
        for (int n : sizes) {
            Problem P = build_problem(synthetic_grid(n), g_natural ? Ordering::NATURAL : Ordering::MULTICOLOR);
            Solution G = gpu_solve(P);
            printf("n=%d colours=%d levels=%d converged=%d newton=%d mismatch=%.2e loop=%.1f ms  bicg per step:", n,
                   P.colors, P.levels, (int)G.converged, G.newton_iters, G.history.back(), G.t.total());
            for (int li : G.linear_iters) printf(" %d", li);
            printf("\n   GPU phases: setup %.1f inject %.1f jacobian %.1f ilu %.1f solve %.1f update %.1f ms\n   mismatch:",
                   G.t.setup, G.t.inject, G.t.jacobian, G.t.ilu, G.t.solve, G.t.update);
            for (double h : G.history) printf(" %.2e", h);
            printf("\n");
            if (getenv("PROBE_WRITE")) write_case(synthetic_grid(n), g_outdir + "/probe" + std::to_string(n) + ".case");
            if (getenv("PROBE_THREADS")) {             // CPU thread-count study (natural ordering)
                Problem Q = build_problem(synthetic_grid(n), Ordering::NATURAL);
                std::stringstream ss(getenv("PROBE_THREADS"));
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    Solution C = cpu::solve(Q, atoi(tok.c_str()));
                    printf("   CPU %s threads: loop=%.1f ms (inject %.1f jac %.1f ilu %.1f solve %.1f)\n", tok.c_str(),
                           C.t.total(), C.t.inject, C.t.jacobian, C.t.ilu, C.t.solve);
                }
            }
            if (getenv("PROBE_CPU")) {
                for (Ordering o : {Ordering::NATURAL, Ordering::MULTICOLOR}) {
                    Problem Q = build_problem(synthetic_grid(n), o);
                    Solution C = cpu::solve(Q, 1);
                    printf("   CPU 1 thread, %s: newton=%d loop=%.1f ms (ilu %.1f, solve %.1f) bicg:",
                           o == Ordering::NATURAL ? "natural   " : "multicolor", C.newton_iters, C.t.total(), C.t.ilu,
                           C.t.solve);
                    for (int li : C.linear_iters) printf(" %d", li);
                    printf("\n");
                }
            }
        }
        return 0;
    }
    if (mode == "solve") {
        Grid g = read_case(casefile);
        Problem P = build_problem(g);
        print_problem(g, P);
        Solution G = gpu_solve(P);
        printf("  GPU: %s in %d Newton iterations, %.1f ms; max mismatch %.2e pu\n",
               G.converged ? "converged" : "NOT converged", G.newton_iters, G.t.total(), max_mismatch(P, G.Vm, G.Va));
        write_solution(G, "solution_gpu.sol");
        return G.converged ? 0 : 1;
    }
    int rc = 0;
    if (mode == "demo" || mode == "all") rc |= run_demo();
    if (mode == "verify" || mode == "all") rc |= run_verify();
    if (mode == "bench" || mode == "all") rc |= run_bench(sizes);
    return rc;
}
