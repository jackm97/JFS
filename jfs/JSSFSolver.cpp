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

    I = SparseMatrix (N*N*3,N*N*3);
    I.setIdentity();
    ADifS = (I - diff * dt * LAPLACEX);
    diffuseSolveS.compute(ADifS);

    projectSolve.compute(LAPLACE);

    b.resize(N*N*3);
    bVec.resize(N*N*2);
    sol.resize(N*N*3);
    solVec.resize(N*N*2);
}

template <class LinearSolver>
JFS_INLINE bool JSSFSolver<LinearSolver>::calcNextStep()
{
    addForce(U, U0, F, dt);
    backstream(U0, U, U, dt, 2);
    diffuse(U, U0, dt, 2);
    projection(U0, U);

    addForce(S, S0, SF, dt);
    backstream(S0, S, U0, dt, 1);
    diffuse(S, S0, dt, 1);
    dissipate(S0, S, dt);
    S = S0;

    satisfyBC(U0);

    return false;
}

template <class LinearSolver>
JFS_INLINE bool JSSFSolver<LinearSolver>::calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources)
{
    bool failedStep = false;

    try
    {
        interpolateForce(forces);
        interpolateSource(sources);

        failedStep = calcNextStep();

        F.setZero();
        SF.setZero();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        failedStep = true;
    }

    if (failedStep) resetFluid();

    return failedStep;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::addForce(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &force, float dt)
{
    dst = src + dt * force ;
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