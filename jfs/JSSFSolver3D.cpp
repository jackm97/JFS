#include <jfs/JSSFSolver3D.h>

#include <jfs/differential_ops/grid_diff3d.h>

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
    grid3D::initializeGrid(N, L, btype, dt);

    this->visc = visc;
    this->diff = diff;
    this->diss = diss;

    this->bound_type_ = ZERO;

    SparseMatrix_ I(N*N*N*3,N*N*N*3);
    I.setIdentity();
    gridDiff3D<SparseMatrix_>::Laplace(this, this->ADifU, 3);
    this->ADifU = (I - visc * dt * this->ADifU);
    this->diffuseSolveU.compute(this->ADifU);

    I = SparseMatrix_ (N*N*N*3,N*N*N*3);
    I.setIdentity();
    gridDiff3D<SparseMatrix_>::Laplace(this, this->ADifS, 1, 3);
    this->ADifS = (I - diff * dt * this->ADifS);
    this->diffuseSolveS.compute(this->ADifS);

    gridDiff3D<SparseMatrix_>::Laplace(this, this->AProject, 1);
    this->projectSolve.compute(this->AProject);

    this->b.resize(N*N*N*3);
    this->bVec.resize(N*N*N*3);

    this->bound_type_ = btype;

    gridDiff3D<SparseMatrix_>::grad(this, this->GRAD);
    gridDiff3D<SparseMatrix_>::div(this, this->DIV);
    
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