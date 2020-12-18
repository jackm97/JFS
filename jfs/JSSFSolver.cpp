#include <jfs/JSSFSolver.h>
#include <iostream>

namespace jfs {

template <class LinearSolver>
JFS_INLINE JSSFSolver<LinearSolver>::JSSFSolver(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc, float diff, float diss)
{
    initialize(N, L, BOUND, dt, visc, diff, diss);
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::initialize(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc, float diff, float diss)
{
    this->visc = visc;
    this->diff = diff;
    this->diss = diss;

    initializeFluid(N, L, BOUND, dt);

    SparseMatrix I(N*N*2,N*N*2);
    I.setIdentity();
    ADifU = (I - visc * dt * VEC_LAPLACE);
    diffuseSolveU.compute(ADifU);
    if (N < 5) std::cout << VEC_LAPLACE.rows() << std::endl;

    I = SparseMatrix (N*N*3,N*N*3);
    I.setIdentity();
    ADifS = (I - diff * dt * LAPLACEX);
    diffuseSolveS.compute(ADifS);
    if (N < 5) std::cout << LAPLACEX.rows() << std::endl;

    projectSolve.compute(LAPLACE);
    if (N < 5) std::cout << LAPLACE.rows() << std::endl;

    b.resize(N*N*3);
    bVec.resize(N*N*2);
    sol.resize(N*N*3);
    solVec.resize(N*N*2);
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::calcNextStep()
{
    addForce(U, U0, F, dt);
    transport(U0, U, U, dt, 2);
    diffuse(U, U0, dt, 2);
    projection(U0, U);

    addForce(S, S0, SF, dt);
    transport(S0, S, U0, dt, 1);
    diffuse(S, S0, dt, 1);
    dissipate(S0, S, dt);
    S = S0;

    satisfyBC(U0);
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources)
{
    interpolateForce(forces);
    interpolateSource(sources);

    calcNextStep();

    F.setZero();
    SF.setZero();
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::addForce(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &force, float dt)
{
    dst = src + dt * force ;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::transport(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt, int dims)
{
    particleTrace(X0, X, u, -dt);

    ij0 = (1/D * X0.array() - .5);

    SparseMatrix *linInterpPtr;
    int fields;

    switch (dims)
    {
    case 1:
        linInterpPtr = &linInterp;
        fields = 3;
        break;

    case 2:
        linInterpPtr = &linInterpVec;
        fields = 1;
        break;
    }

    calcLinInterp(*linInterpPtr, ij0, dims, fields);

    dst = *linInterpPtr * src;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::particleTrace(Eigen::VectorXf &X0, const Eigen::VectorXf &X, const Eigen::VectorXf &u, float dt)
{
    ij0 = 1/D * ( (X + 1/2 * (dt * X)) + dt/2 * u ).array() - .5;
    
    calcLinInterp(linInterpVec, ij0, 2);
    X0 = X + dt * ( linInterpVec * u );
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::diffuse(Eigen::VectorXf &dst, const Eigen::VectorXf &src, float dt, int dims)
{
    switch (dims)
    {
    case 1:
        dst = (diffuseSolveS).solve(src);
        break;
    
    case 2:
        dst = (diffuseSolveU).solve(src);
        break;
    }
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::projection(Eigen::VectorXf &dst, const Eigen::VectorXf &src)
{
    bVec = (DIV * src);

    solVec = projectSolve.solve(bVec);

    dst = src - GRAD * solVec;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::dissipate(Eigen::VectorXf &dst, const Eigen::VectorXf &src, float dt)
{
    dst = 1/(1 + dt * diss) * src;
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class JSSFSolver<>;
template class JSSFSolver<fastZeroSolver>;
template class JSSFSolver<iterativeSolver>;
#endif

} // namespace jfs