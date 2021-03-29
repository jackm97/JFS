#include <jfs/JSSFSolver3D.h>
#include <iostream>

namespace jfs {

template <class LinearSolver, int StorageOrder>
JFS_INLINE JSSFSolver3D<LinearSolver, StorageOrder>::
JSSFSolver3D(unsigned int N, float L, BoundType btype, float dt, float visc, float diff, float diss)
{
    initialize(N, L, btype, dt, visc, diff, diss);
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolver3D<LinearSolver, StorageOrder>::
initialize(unsigned int N, float L, BoundType btype, float dt, float visc, float diff, float diss)
{
    using grid3D = grid3D<StorageOrder>;
    grid3D::initializeGrid(N, L, btype, dt);

    this->visc = visc;
    this->diff = diff;
    this->diss = diss;

    this->bound_type_ = ZERO;

    SparseMatrix_ I(N*N*N*3,N*N*N*3);
    I.setIdentity();
    grid3D::Laplace(this->ADifU, 3);
    this->ADifU = (I - visc * dt * this->ADifU);
    this->diffuseSolveU.compute(this->ADifU);

    I = SparseMatrix_ (N*N*N*3,N*N*N*3);
    I.setIdentity();
    grid3D::Laplace(this->ADifS, 1, 3);
    this->ADifS = (I - diff * dt * this->ADifS);
    this->diffuseSolveS.compute(this->ADifS);

    grid3D::Laplace(this->AProject, 1);
    this->projectSolve.compute(this->AProject);

    this->b.resize(N*N*N*3);
    this->bVec.resize(N*N*N*3);

    this->bound_type_ = btype;

    grid3D::grad(this->GRAD);
    grid3D::div(this->DIV);
    
    this->U.resize(N*N*N*3);

    this->F.resize(N*N*N*3);
    
    this->S.resize(N*N*N*3);
    this->SF.resize(N*N*N*3);

    this->resetFluid();
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class JSSFSolver3D<>;
template class JSSFSolver3D<fastZeroSolver>;
template class JSSFSolver3D<iterativeSolver>;
#endif

} // namespace jfs