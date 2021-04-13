#include "JSSFSolver.h"
#include <jfs/differential_ops/grid_diff2d.h>

#include <iostream>

namespace jfs {

template <class LinearSolver, int StorageOrder>
JFS_INLINE JSSFSolver<LinearSolver, StorageOrder>::
JSSFSolver(unsigned int N, float L, BoundType btype, float dt, float visc)
{
    initialize(N, L, btype, dt, visc);
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolver<LinearSolver, StorageOrder>::
initialize(unsigned int N, float L, BoundType btype, float dt, float visc)
{
    grid2D::initializeGrid(N, L, btype, dt);

    this->visc = visc;

    this->bound_type_ = ZERO;

    SparseMatrix_ I(N*N*2,N*N*2);
    I.setIdentity();
    gridDiff2D<SparseMatrix_>::Laplace(this, this->ADifU, 2);
    this->ADifU = (I - visc * dt * this->ADifU);
    this->diffuseSolveU.compute(this->ADifU);

    gridDiff2D<SparseMatrix_>::Laplace(this, this->AProject, 1);
    this->projectSolve.compute(this->AProject);

    this->b.resize(N*N*3);
    this->bVec.resize(N*N*2);

    this->bound_type_ = btype;

    gridDiff2D<SparseMatrix_>::grad(this, this->GRAD);
    gridDiff2D<SparseMatrix_>::div(this, this->DIV);

    this->U.resize(N * N * 2);

    this->F.resize(N * N * 2);

    this->resetFluid();
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolver<LinearSolver, StorageOrder>::resetFluid()
{
    U.setZero();
    U0 = U;

    F.setZero();
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE bool JSSFSolver<LinearSolver, StorageOrder>::calcNextStep()
{
    auto dt = this->dt;
    
    addForce(U, U0, F, dt);
    this->backstream(U0.data(), U.data(), U.data(), dt, VECTOR_FIELD);
    diffuse(U, U0, dt, VECTOR_FIELD);
    projection(U0, U);

    this->satisfyBC(U0.data(), VECTOR_FIELD, 1);

    return false;
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE bool JSSFSolver<LinearSolver, StorageOrder>::
calcNextStep(const std::vector<Force> forces)
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

        failedStep = calcNextStep();

        F.setZero();
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
JFS_INLINE void JSSFSolver<LinearSolver, StorageOrder>::
addForce(Vector_ &dst, const Vector_ &src, const Vector_ &force, float dt)
{
    dst = src + dt * force ;
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolver<LinearSolver, StorageOrder>::
projection(Vector_ &dst, const Vector_ &src)
{
    static Vector_ x;
    bVec = (DIV * src);

    x = projectSolve.solve(bVec);

    dst = src - GRAD * x;
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolver<LinearSolver, StorageOrder>::
diffuse(Vector_ &dst, const Vector_ &src, float dt, FieldType ftype)
{
    switch (ftype)
    {
    case SCALAR_FIELD:
        break;
    
    case VECTOR_FIELD:
        dst = (diffuseSolveU).solve(src);
        break;
    }
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class JSSFSolver<>;
template class JSSFSolver<fastZeroSolver>;
template class JSSFSolver<iterativeSolver>;
#endif

} // namespace jfs