#include <jfs/JSSFSolver3D.h>
#include <iostream>

namespace jfs {

template <class LinearSolver, int StorageOrder>
JFS_INLINE JSSFSolver3D<LinearSolver, StorageOrder>::
JSSFSolver3D(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc, float diff, float diss)
{
    initialize(N, L, BOUND, dt, visc, diff, diss);
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolver3D<LinearSolver, StorageOrder>::
initialize(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc, float diff, float diss)
{
    using grid3D = grid3D<StorageOrder>;
    grid3D::initializeGrid(N, L, BOUND, dt);

    this->visc = visc;
    this->diff = diff;
    this->diss = diss;

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

    grid3D::grad(this->GRAD);
    grid3D::div(this->DIV);
    
    this->U.resize(N*N*N*3);

    this->F.resize(N*N*N*3);
    
    this->S.resize(N*N*N*3);
    this->SF.resize(N*N*N*3);

    this->resetFluid();
}

template <class LinearSolver, int StorageOrder>
JFS_INLINE void JSSFSolver3D<LinearSolver, StorageOrder>::getImage(Eigen::VectorXf &image)
{
    using grid3D = grid3D<StorageOrder>;
    
    auto BOUND = grid3D::BOUND;
    auto L = grid3D::L;
    auto N = grid3D::N;
    auto D = grid3D::D;

    if (image.rows() != N*N*N*3)
        image.resize(N*N*N*3);

    for (int i=0; i < N; i++)
        for (int j=0; j < N; j++)
            for (int k=0; k < N; k++)
        {
            image(N*3*N*k + N*3*j + i*3 + 0) = this->S(0*N*N*N + N*N*k + N*j + i);
            image(N*3*N*k + N*3*j + i*3 + 1) = this->S(1*N*N*N + N*N*k + N*j + i);
            image(N*3*N*k + N*3*j + i*3 + 2) = this->S(2*N*N*N + N*N*k + N*j + i);
        }
    image = (image.array() <= 1.).select(image, 1.);
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class JSSFSolver3D<>;
template class JSSFSolver3D<fastZeroSolver>;
template class JSSFSolver3D<iterativeSolver>;
#endif

} // namespace jfs