#include <jfs/JSSFSolver.h>
#include <iostream>

namespace jfs {

template <class LinearSolver, int StorageOrder>
JFS_INLINE JSSFSolver<LinearSolver, StorageOrder>::
JSSFSolver(unsigned int N, float L, BoundType btype, float dt, float visc, float diff, float diss)
{
    initialize(N, L, btype, dt, visc, diff, diss);
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolver<LinearSolver, StorageOrder>::
initialize(unsigned int N, float L, BoundType btype, float dt, float visc, float diff, float diss)
{
    using grid2D = grid2D<StorageOrder>;
    grid2D::initializeGrid(N, L, btype, dt);

    this->visc = visc;
    this->diff = diff;
    this->diss = diss;

    this->bound_type_ = ZERO;

    SparseMatrix_ I(N*N*2,N*N*2);
    I.setIdentity();
    grid2D::Laplace(this->ADifU, 2);
    this->ADifU = (I - visc * dt * this->ADifU);
    this->diffuseSolveU.compute(this->ADifU);

    I = SparseMatrix_ (N*N*3,N*N*3);
    I.setIdentity();
    grid2D::Laplace(this->ADifS, 1, 3);
    this->ADifS = (I - diff * dt * this->ADifS);
    this->diffuseSolveS.compute(this->ADifS);

    grid2D::Laplace(this->AProject, 1);
    this->projectSolve.compute(this->AProject);

    this->b.resize(N*N*3);
    this->bVec.resize(N*N*2);

    this->bound_type_ = btype;

    grid2D::grad(this->GRAD);
    grid2D::div(this->DIV);

    this->U.resize(N*N*2);

    this->F.resize(N*N*2);
    
    this->S.resize(N*N*3);
    this->SF.resize(N*N*3);

    this->resetFluid();
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class JSSFSolver<>;
template class JSSFSolver<fastZeroSolver>;
template class JSSFSolver<iterativeSolver>;
#endif

} // namespace jfs