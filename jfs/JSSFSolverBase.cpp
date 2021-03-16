#include <jfs/JSSFSolverBase.h>
#include <iostream>

namespace jfs {

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolverBase<LinearSolver, StorageOrder>::resetFluid()
{
    U.setZero();
    U0 = U;

    F.setZero();

    S.setZero();
    S0 = S;
    SF.setZero();
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE bool JSSFSolverBase<LinearSolver, StorageOrder>::calcNextStep()
{
    auto dt = this->dt;
    
    addForce(U, U0, F, dt);
    this->backstream(U0, U, U, dt, VECTOR_FIELD);
    diffuse(U, U0, dt, VECTOR_FIELD);
    projection(U0, U);

    addForce(S, S0, SF, dt);
    this->backstream(S0, S, U0, dt, SCALAR_FIELD, 3);
    diffuse(S, S0, dt, SCALAR_FIELD);
    dissipate(S0, S, dt);
    S = S0;

    this->satisfyBC(U0, VECTOR_FIELD, 1);

    return false;
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE bool JSSFSolverBase<LinearSolver, StorageOrder>::
calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources)
{
    bool failedStep = false;

    try
    {
        this->interpolateForce(forces, F);
        this->interpolateSource(sources, SF);

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

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolverBase<LinearSolver, StorageOrder>::
addForce(Vector_ &dst, const Vector_ &src, const Vector_ &force, float dt)
{
    dst = src + dt * force ;
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolverBase<LinearSolver, StorageOrder>::
projection(Vector_ &dst, const Vector_ &src)
{
    static Vector_ x;
    bVec = (DIV * src);

    x = projectSolve.solve(bVec);

    dst = src - GRAD * x;
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolverBase<LinearSolver, StorageOrder>::
diffuse(Vector_ &dst, const Vector_ &src, float dt, FieldType ftype)
{
    switch (ftype)
    {
    case SCALAR_FIELD:
        dst = (diffuseSolveS).solve(src);
        break;
    
    case VECTOR_FIELD:
        dst = (diffuseSolveU).solve(src);
        break;
    }
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolverBase<LinearSolver, StorageOrder>::
dissipate(Vector_ &dst, const Vector_ &src, float dt)
{
    dst = 1/(1 + dt * diss) * src;
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class JSSFSolverBase<genSolver, Eigen::ColMajor>;
template class JSSFSolverBase<fastZeroSolver, Eigen::ColMajor>;
template class JSSFSolverBase<iterativeSolver, Eigen::ColMajor>;
#endif

} // namespace jfs