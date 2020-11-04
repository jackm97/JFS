#include <jfs/JSSFSolver.h>

namespace jfs {

template <class LinearSolver>
JFS_INLINE JSSFSolver<LinearSolver>::JSSFSolver(){}

template <class LinearSolver>
JFS_INLINE JSSFSolver<LinearSolver>::JSSFSolver(unsigned int N, float L, float D, BOUND_TYPE BOUND, float dt)
{
    initialize(N, L, D, BOUND, dt);
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::initialize(unsigned int N, float L, float D, BOUND_TYPE BOUND, float dt)
{
    this->N = N;
    this->L = L;
    this->D = D;
    this->BOUND = BOUND;
    this->dt = dt;

    U.resize(N*N*2);
    U.setZero();
    U0 = U;
    UTemp = U;
    S.resize(N*N);
    S.setZero();
    S0 = S;
    STemp = S;
    X.resize(N*N*2);
    setXGrid2D();
    X0 = X;
    XTemp = X;

    Laplace2D(LAPLACE, 1);
    Laplace2D(VEC_LAPLACE, 2);
    div2D(DIV);
    grad2D(GRAD);

    projectSolve.compute(LAPLACE);

    Eigen::SparseMatrix<float> I(N*N*2,N*N*2), A(N*N*2,N*N*2);
    I.setIdentity();
    A = (I - 0. * dt * VEC_LAPLACE);
    diffuseSolveU.compute(A);

    I = Eigen::SparseMatrix<float> (N*N,N*N);
    I.setIdentity();
    A = (I - 0. * dt * LAPLACE);
    diffuseSolveS.compute(A);

    b.resize(N*N);
    bVec.resize(N*N*2);
    sol.resize(N*N);
    solVec.resize(N*N*2);

    k1.resize(N*N*2);
    k2.resize(N*N*2);

    ij0.resize(N*N*2);
    linInterp.resize(N*N,N*N);
    linInterpVec.resize(N*N*2,N*N*2);
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::calcNextStep(const Eigen::SparseMatrix<float> &force, const Eigen::SparseMatrix<float> &source, float dt)
{
    addForce(U, U0, force, dt);
    transport2D(U0, U, U, dt, 2);
    diffuse2D(U, U0, dt, 2);
    projection2D(U0, U);

    addForce(S, S0, source, dt);
    transport2D(S0, S, U0, dt, 1);
    diffuse2D(S, S0, dt, 1);
    dissipate2D(S0, S, dt);

    satisfyBC(U0);
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::setXGrid2D()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int dim = 0;
            X(dim*N*N + j*N + i) = D*(i+.5);
            
            dim = 1;
            X(dim*N*N + j*N + i) = D*(j+.5);
        }
    }
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::addForce(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &force, float dt)
{
    dst = src + dt * force ;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::transport2D(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt, int dims)
{
    particleTrace(X0, X, u, -dt);

    ij0 = (1/D * X0.array() - .5);

    Eigen::SparseMatrix<float> *linInterpPtr;

    switch (dims)
    {
    case 1:
        linInterpPtr = &linInterp;
        break;

    case 2:
        linInterpPtr = &linInterpVec;
        break;
    }

    calcLinInterp2D(*linInterpPtr, ij0, dims);

    dst = *linInterpPtr * src;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::particleTrace(Eigen::VectorXf &X0, const Eigen::VectorXf &X, const Eigen::VectorXf &u, float dt)
{
    ij0 = 1/D * ( (X + 1/2 * (dt * X)) + dt/2 * u ).array() - .5;
    
    calcLinInterp2D(linInterpVec, ij0, 2);
    X0 = X + dt * ( linInterpVec * u );
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::diffuse2D(Eigen::VectorXf &dst, const Eigen::VectorXf &src, float dt, int dims)
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
JFS_INLINE void JSSFSolver<LinearSolver>::projection2D(Eigen::VectorXf &dst, const Eigen::VectorXf &src)
{
    bVec = (DIV * src);

    solVec = projectSolve.solve(bVec);

    dst = src - GRAD * solVec;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::dissipate2D(Eigen::VectorXf &dst, const Eigen::VectorXf &src, float dt)
{
    dst = 1/(1 + dt * 0.05) * src;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::satisfyBC(Eigen::VectorXf &u)
{
    if (BOUND == PERIODIC) return;

    int i,j;
    for (int idx=0; idx < N; idx++)
    {
        // top
        i = idx;
        j = N-1;
        u(N*N*1 + N*j + i) = 0;

        // bottom
        i = idx;
        j = 0;
        u(N*N*1 + N*j + i) = 0;

        // left
        j = idx;
        i = N-1;
        u(N*N*0 + N*j + i) = 0;

        // right
        j = idx;
        i = 0;
        u(N*N*0 + N*j + i) = 0;
    }
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::Laplace2D(Eigen::SparseMatrix<float> &dst, unsigned int dims)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*5*dims);

    for (int dim = 0; dim < dims; dim++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int iMat = dim*N*N + j*N + i;
                int jMat = iMat;
                
                // x,y
                tripletList.push_back(T(iMat,jMat,-4.f));
                
                //x,y+h
                jMat = dim*N*N + (j+1)*N + i;
                if ((j+1) >= N)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = dim*N*N + 1*N + i;  
                            tripletList.push_back(T(iMat,jMat,1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,1.f));
                
                //x,y-h
                jMat = dim*N*N + (j-1)*N + i;
                if ((j-1) < 0)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = dim*N*N + (N-2)*N + i;  
                            tripletList.push_back(T(iMat,jMat,1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,1.f));
                
                //x+h,y
                jMat = dim*N*N + j*N + (i+1);
                if ((i+1) >= N)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = dim*N*N + j*N + 1;  
                            tripletList.push_back(T(iMat,jMat,1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,1.f));
                
                //x-h,y
                jMat = dim*N*N + j*N + i-1;
                if ((i-1) < 0)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = dim*N*N + j*N + N-2;  
                            tripletList.push_back(T(iMat,jMat,1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,1.f));
            }
        }
    }

    dst = Eigen::SparseMatrix<float>(N*N*dims,N*N*dims);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(D*D) * dst;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::div2D(Eigen::SparseMatrix<float> &dst)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*2*2);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int iMat = j*N + i;
            int jMat = iMat;

            int dim = 0;
            
            //x+h,y
            jMat = dim*N*N + j*N + (i+1);
            if ((i+1) >= N)
                switch (BOUND)
                {
                    case ZERO:
                        break;

                    case PERIODIC:
                        jMat = dim*N*N + j*N + 1;  
                        tripletList.push_back(T(iMat,jMat,1.f)); 
                }
            else
                tripletList.push_back(T(iMat,jMat,1.f));
            
            //x-h,y
            jMat = dim*N*N + j*N + i-1;
            if ((i-1) < 0)
                switch (BOUND)
                {
                    case ZERO:
                        break;

                    case PERIODIC:
                        jMat = dim*N*N + j*N + N-2;  
                        tripletList.push_back(T(iMat,jMat,-1.f)); 
                }
            else
                tripletList.push_back(T(iMat,jMat,-1.f));

            dim = 1;
            
            //x,y+h
            jMat = dim*N*N + (j+1)*N + i;
            if ((j+1) >= N)
                switch (BOUND)
                {
                    case ZERO:
                        break;

                    case PERIODIC:
                        jMat = dim*N*N + 1*N + i;  
                        tripletList.push_back(T(iMat,jMat,1.f)); 
                }
            else
                tripletList.push_back(T(iMat,jMat,1.f));
            
            //x,y-h
            jMat = dim*N*N + (j-1)*N + i;
            if ((j-1) < 0)
                switch (BOUND)
                {
                    case ZERO:
                        break;

                    case PERIODIC:
                        jMat = dim*N*N + (N-2)*N + i;  
                        tripletList.push_back(T(iMat,jMat,-1.f)); 
                }
            else
                tripletList.push_back(T(iMat,jMat,-1.f));
        }
    }

    dst = Eigen::SparseMatrix<float>(N*N,N*N*2);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::grad2D(Eigen::SparseMatrix<float> &dst)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*2*2);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int iMat = j*N + i;
            int jMat = iMat;

            int dim = 0;
            iMat = dim*N*N + j*N + i;
            
            //x+h,y
            jMat = j*N + (i+1);
            if ((i+1) >= N)
                switch (BOUND)
                {
                    case ZERO:
                        break;

                    case PERIODIC:
                        jMat = j*N + 1;  
                        tripletList.push_back(T(iMat,jMat,1.f)); 
                }
            else
                tripletList.push_back(T(iMat,jMat,1.f));
            
            //x-h,y
            jMat = j*N + i-1;
            if ((i-1) < 0)
                switch (BOUND)
                {
                    case ZERO:
                        break;

                    case PERIODIC:
                        jMat = j*N + N-2;  
                        tripletList.push_back(T(iMat,jMat,-1.f)); 
                }
            else
                tripletList.push_back(T(iMat,jMat,-1.f));

            dim = 1;
            iMat = dim*N*N + j*N + i;
            
            //x,y+h
            jMat = (j+1)*N + i;
            if ((j+1) >= N)
                switch (BOUND)
                {
                    case ZERO:
                        break;

                    case PERIODIC:
                        jMat = 1*N + i;  
                        tripletList.push_back(T(iMat,jMat,1.f)); 
                }
            else
                tripletList.push_back(T(iMat,jMat,1.f));
            
            //x,y-h
            jMat = (j-1)*N + i;
            if ((j-1) < 0)
                switch (BOUND)
                {
                    case ZERO:
                        break;

                    case PERIODIC:
                        jMat = (N-2)*N + i;  
                        tripletList.push_back(T(iMat,jMat,-1.f)); 
                }
            else
                tripletList.push_back(T(iMat,jMat,-1.f));
        }
    }

    dst = Eigen::SparseMatrix<float>(N*N*2,N*N);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}

template <class LinearSolver>
JFS_INLINE void JSSFSolver<LinearSolver>::calcLinInterp2D(Eigen::SparseMatrix<float> &dst, const Eigen::VectorXf &ij0, int dims)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*4*dims);
    
    for (int x = 0; x < N; x++)
    {
        for (int y = 0; y < N; y++)
        {
            float i0 = ij0(0*N*N + N*y + x);
            float j0 = ij0(1*N*N + N*y + x);

            switch (BOUND)
            {
                case ZERO:
                    i0 = (i0 < 0) ? 0:i0;
                    i0 = (i0 > (N-1)) ? (N-1):i0;
                    j0 = (j0 < 0) ? 0:j0;
                    j0 = (j0 > (N-1)) ? (N-1):j0;
                    break;
                
                case PERIODIC:
                    i0 = (i0 < 0) ? (N-1+i0):i0;
                    i0 = (i0 > (N-1)) ? (i0 - (N-1)):i0;
                    j0 = (j0 < 0) ? (N-1+j0):j0;
                    j0 = (j0 > (N-1)) ? (j0 - (N-1)):j0;
                    break;
            }

            int i0_floor = (int) i0;
            int j0_floor = (int) j0;
            int i0_ceil = i0_floor + 1;
            int j0_ceil = j0_floor + 1;
            int iMat, jMat, iref, jref;
            float part;

            for (int dim = 0; dim < dims; dim++)
            {
                iMat = dim*N*N + y*N + x;
                
                jMat = dim*N*N + j0_floor*N + i0_floor;
                part = (i0_ceil - i0)*(j0_ceil - j0);
                tripletList.push_back(T(iMat, jMat, std::abs(part)));
                
                jMat = dim*N*N + j0_ceil*N + i0_floor;
                if (j0_ceil < N)
                {
                    part = (i0_ceil - i0)*(j0_floor - j0);
                    tripletList.push_back(T(iMat, jMat, std::abs(part)));
                }
                
                jMat = dim*N*N + j0_floor*N + i0_ceil;
                if (i0_ceil < N)
                {
                    part = (i0_floor - i0)*(j0_ceil - j0);
                    tripletList.push_back(T(iMat, jMat, std::abs(part)));
                }
                
                jMat = dim*N*N + j0_ceil*N + i0_ceil;
                if (i0_ceil < N && j0_ceil < N)
                {
                    part = (i0_floor - i0)*(j0_floor - j0);
                    tripletList.push_back(T(iMat, jMat, std::abs(part)));
                }
            }
        }
        
    }

    dst = Eigen::SparseMatrix<float>(N*N*dims,N*N*dims);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class JSSFSolver<>;
template class JSSFSolver<fastZeroSolver>;
#endif

} // namespace jfs