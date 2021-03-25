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
    this->backstream(U0.data(), U.data(), U.data(), dt, VECTOR_FIELD);
    diffuse(U, U0, dt, VECTOR_FIELD);
    projection(U0, U);

    addForce(S, S0, SF, dt);
    this->backstream(S0.data(), S.data(), U0.data(), dt, SCALAR_FIELD, 3);
    diffuse(S, S0, dt, SCALAR_FIELD);
    dissipate(S0, S, dt);
    S = S0;

    this->satisfyBC(U0.data(), VECTOR_FIELD, 1);

    return false;
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE bool JSSFSolverBase<LinearSolver, StorageOrder>::
calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources)
{
    bool failedStep = false;

    try
    {
        for (int i = 0; i < forces.size(); i++)
        {
            float force[3] = {
                forces[i].force[0],
                forces[i].force[1],
                forces[i].force[2]
            };
            float point[3] = {
                forces[i].pos[0]/this->D,
                forces[i].pos[1]/this->D,
                forces[i].pos[2]/this->D
            };
            this->interpPointToGrid(force, point, F.data(), VECTOR_FIELD, 1, Add);
        }
        for (int i = 0; i < sources.size(); i++)
        {
            float source[3] = {
                sources[i].color[0] * sources[i].strength,
                sources[i].color[1] * sources[i].strength,
                sources[i].color[2] * sources[i].strength
            };
            float point[3] = {
                sources[i].pos[0]/this->D,
                sources[i].pos[1]/this->D,
                sources[i].pos[2]/this->D
            };
            this->interpPointToGrid(source, point, SF.data(), SCALAR_FIELD, 3, Add);
        }

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