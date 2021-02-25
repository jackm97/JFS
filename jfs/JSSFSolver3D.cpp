#include <jfs/JSSFSolver3D.h>
#include <iostream>

namespace jfs {

template <class LinearSolver>
JFS_INLINE JSSFSolver3D<LinearSolver>::JSSFSolver3D(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc, float diff, float diss)
{
    initialize(N, L, BOUND, dt, visc, diff, diss);
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver3D<LinearSolver>::initialize(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc, float diff, float diss)
{
    this->visc = visc;
    this->diff = diff;
    this->diss = diss;

    initializeFluid(N, L, BOUND, dt);

    SparseMatrix I(N*N*N*3,N*N*N*3);
    I.setIdentity();
    Laplace(ADifU, 3);
    ADifU = (I - visc * dt * ADifU);
    diffuseSolveU.compute(ADifU);

    I = SparseMatrix (N*N*N*3,N*N*N*3);
    I.setIdentity();
    Laplace(ADifS, 1, 3);
    ADifS = (I - diff * dt * ADifS);
    diffuseSolveS.compute(ADifS);

    Laplace(AProject, 1);
    projectSolve.compute(AProject);

    b.resize(N*N*N*3);
    bVec.resize(N*N*N*3);

    grad(GRAD);
    div(DIV);
}

template <class LinearSolver>
JFS_INLINE bool JSSFSolver3D<LinearSolver>::calcNextStep()
{
    addForce(U, U0, F, dt);
    backstream(U0, U, U, dt, 3);
    diffuse(U, U0, dt, 3);
    projection(U0, U);

    addForce(S, S0, SF, dt);
    backstream(S0, S, U0, dt, 1, 3);
    diffuse(S, S0, dt, 1);
    dissipate(S0, S, dt);
    S = S0;

    satisfyBC(U0);

    return false;
}

template <class LinearSolver>
JFS_INLINE bool JSSFSolver3D<LinearSolver>::calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources)
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
JFS_INLINE void JSSFSolver3D<LinearSolver>::addForce(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &force, float dt)
{
    dst = src + dt * force ;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver3D<LinearSolver>::diffuse(Eigen::VectorXf &dst, const Eigen::VectorXf &src, float dt, int dims)
{
    switch (dims)
    {
    case 1:
        dst = (diffuseSolveS).solve(src);
        break;
    
    case 3:
        dst = (diffuseSolveU).solve(src);
        break;
    }
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver3D<LinearSolver>::projection(Eigen::VectorXf &dst, const Eigen::VectorXf &src)
{
    static Eigen::VectorXf x;
    bVec = (DIV * src);

    x = projectSolve.solve(bVec);

    dst = src - GRAD * x;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver3D<LinearSolver>::dissipate(Eigen::VectorXf &dst, const Eigen::VectorXf &src, float dt)
{
    dst = 1/(1 + dt * diss) * src;
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class JSSFSolver3D<>;
template class JSSFSolver3D<fastZeroSolver>;
template class JSSFSolver3D<iterativeSolver>;
#endif

} // namespace jfs