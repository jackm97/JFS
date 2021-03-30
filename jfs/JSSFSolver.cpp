#include <jfs/JSSFSolver.h>

#include <jfs/differential_ops/grid_diff2d.h>

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

    this->U.resize(N*N*2);

    this->F.resize(N*N*2);

    this->resetFluid();
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class JSSFSolver<>;
template class JSSFSolver<fastZeroSolver>;
template class JSSFSolver<iterativeSolver>;
#endif

} // namespace jfs