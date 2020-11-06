#include <jfs/JSSFSolver.h>

namespace jfs {

template <class LinearSolver>
JFS_INLINE JSSFSolver<LinearSolver>::JSSFSolver(){}

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

    projectSolve.compute(LAPLACE);

    Eigen::SparseMatrix<float> I(N*N*2,N*N*2), A(N*N*2,N*N*2);
    I.setIdentity();
    A = (I - visc * dt * VEC_LAPLACE);
    diffuseSolveU.compute(A);

    I = Eigen::SparseMatrix<float> (N*N*3,N*N*3);
    I.setIdentity();
    A = (I - diff * dt * LAPLACE3);
    diffuseSolveS.compute(A);

    b.resize(N*N*3);
    bVec.resize(N*N*2);
    sol.resize(N*N*3);
    solVec.resize(N*N*2);
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::calcNextStep()
{
    addForce(U, U0, F, dt);
    transport(U0, U, U, dt, 2);
    diffuse(U, U0, dt, 2);
    projection(U0, U);

    addForce(S, S0, SF, dt);
    transport(S0, S, U0, dt, 1);
    diffuse(S, S0, dt, 1);
    dissipate(S0, S, dt);
    S = S0;

    satisfyBC(U0);
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::calcNextStep(const std::vector<Force2D> forces, const std::vector<Source2D> sources)
{
    interpolateForce(forces);
    interpolateSource(sources);

    calcNextStep();

    F.setZero();
    FTemp.setZero();
    SF.setZero();
    SFTemp.setZero();
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::getImage(Eigen::VectorXf &image)
{
    if (image.rows() != N*N*3)
        image.resize(N*N*3);

    for (int i=0; i < N; i++)
        for (int j=0; j < N; j++)
        {
            image(N*3*j + 0 + i*3) = S(0*N*N + N*j + i);
            image(N*3*j + 1 + i*3) = S(1*N*N + N*j + i);
            image(N*3*j + 2 + i*3) = S(2*N*N + N*j + i);
        }
    image = (image.array() <= 1.).select(image, 1.);
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::addForce(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &force, float dt)
{
    dst = src + dt * force ;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::transport(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt, int dims)
{
    particleTrace(X0, X, u, -dt);

    ij0 = (1/D * X0.array() - .5);

    Eigen::SparseMatrix<float> *linInterpPtr;
    int fields;

    switch (dims)
    {
    case 1:
        linInterpPtr = &linInterp;
        fields = 3;
        break;

    case 2:
        linInterpPtr = &linInterpVec;
        fields = 1;
        break;
    }

    calcLinInterp(*linInterpPtr, ij0, dims, fields);

    dst = *linInterpPtr * src;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::particleTrace(Eigen::VectorXf &X0, const Eigen::VectorXf &X, const Eigen::VectorXf &u, float dt)
{
    ij0 = 1/D * ( (X + 1/2 * (dt * X)) + dt/2 * u ).array() - .5;
    
    calcLinInterp(linInterpVec, ij0, 2);
    X0 = X + dt * ( linInterpVec * u );
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::diffuse(Eigen::VectorXf &dst, const Eigen::VectorXf &src, float dt, int dims)
{
    LinearSolver* LinSolvePtr; 
    switch (dims)
    {
    case 1:
        LinSolvePtr = &diffuseSolveS;
        break;
    
    case 2:
        LinSolvePtr = &diffuseSolveU;
        break;
    }

    dst = (*LinSolvePtr).solve(src);
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
#endif

} // namespace jfs