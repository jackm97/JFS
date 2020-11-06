#include <jfs/fluid2D.h>

namespace jfs {


JFS_INLINE void fluid2D::initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dt)
{
    this->N = N;
    this->L = L;
    this->D = L/(N-1);
    this->BOUND = BOUND;
    this->dt = dt;

    U.resize(N*N*2);
    X.resize(N*N*2);

    F.resize(N*N*2);
    FTemp.resize(N*N*2);
    
    S.resize(3*N*N);
    SF.resize(3*N*N);
    SFTemp.resize(3*N*N);

    Laplace(LAPLACE, 1);
    Laplace(LAPLACE3, 1, 3);
    Laplace(VEC_LAPLACE, 2);
    div(DIV);
    div(DIV3,3);
    grad(GRAD);

    ij0.resize(N*N*2);
    linInterp.resize(N*N,N*N);
    linInterpVec.resize(N*N*2,N*N*2);

    resetFluid();
}


JFS_INLINE void fluid2D::resetFluid()
{
    U.setZero();
    U0 = U;
    UTemp = U;
    setXGrid();
    X0 = X;
    XTemp = X;

    F.setZero();
    FTemp.setZero();

    S.setZero();
    S0 = S;
    STemp = S;
    SF.setZero();
    SFTemp.setZero();
}


JFS_INLINE void fluid2D::setXGrid()
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


JFS_INLINE void fluid2D::interpolateForce(const std::vector<Force2D> forces)
{
    for (int f=0; f < forces.size(); f++)
    {
        const Force2D &force = forces[f];
        if (force.x>L || force.y>L) continue;
        
        float i = force.x/D;
        float j = force.y/D;

        int i0 = std::floor(i);
        int j0 = std::floor(j);

        int dims = 2;
        float fArr[2] = {force.Fx, force.Fy};
        for (int dim=0; dim < dims; dim++)
        {
            FTemp(N*N*dim + N*j0 + i0) += fArr[dim]*std::abs((j0+1 - j)*(i0+1 - i));
            FTemp(N*N*dim + N*j0 + (i0+1)) += fArr[dim]*std::abs((j0+1 - j)*(i0 - i));
            FTemp(N*N*dim + N*(j0+1) + i0) += fArr[dim]*std::abs((j0 - j)*(i0+1 - i));
            FTemp(N*N*dim + N*(j0+1) + (i0+1)) += fArr[dim]*std::abs((j0 - j)*(i0 - i));
        }
    }
    F = FTemp.sparseView();
}


JFS_INLINE void fluid2D::interpolateSource(const std::vector<Source2D> sources)
{
    for (int f=0; f < sources.size(); f++)
    {
        const Source2D &source = sources[f];
        if (source.x>L || source.y>L) continue;
        
        float i = source.x/D;
        float j = source.y/D;

        int i0 = std::floor(i);
        int j0 = std::floor(j);

        int dims = 1;
        for (int c=0; c < 3; c++)
        {
            float fArr[1] = {source.color(c) * source.strength};
            for (int dim=0; dim < dims; dim++)
            {
                SFTemp(c*N*N*dims + N*N*dim + N*j0 + i0) += fArr[dim]*std::abs((j0+1 - j)*(i0+1 - i));
                SFTemp(c*N*N*dims + N*N*dim + N*j0 + (i0+1)) += fArr[dim]*std::abs((j0+1 - j)*(i0 - i));
                SFTemp(c*N*N*dims + N*N*dim + N*(j0+1) + i0) += fArr[dim]*std::abs((j0 - j)*(i0+1 - i));
                SFTemp(c*N*N*dims + N*N*dim + N*(j0+1) + (i0+1)) += fArr[dim]*std::abs((j0 - j)*(i0 - i));
            }
            SF = SFTemp.sparseView();
        }
    }
}


JFS_INLINE void fluid2D::satisfyBC(Eigen::VectorXf &u)
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


JFS_INLINE void fluid2D::Laplace(Eigen::SparseMatrix<float> &dst, unsigned int dims, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*5*dims);

    for (int field = 0; field < fields; field++)
    {
        for (int dim = 0; dim < dims; dim++)
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    int iMat = field*dims*N*N + dim*N*N + j*N + i;
                    int jMat = iMat;
                    
                    // x,y
                    tripletList.push_back(T(iMat,jMat,-4.f));
                    
                    //x,y+h
                    jMat = field*dims*N*N + dim*N*N + (j+1)*N + i;
                    if ((j+1) >= N)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N + dim*N*N + 1*N + i;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x,y-h
                    jMat = field*dims*N*N + dim*N*N + (j-1)*N + i;
                    if ((j-1) < 0)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N + dim*N*N + (N-2)*N + i;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x+h,y
                    jMat = field*dims*N*N + dim*N*N + j*N + (i+1);
                    if ((i+1) >= N)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N + dim*N*N + j*N + 1;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x-h,y
                    jMat = field*dims*N*N + dim*N*N + j*N + i-1;
                    if ((i-1) < 0)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N + dim*N*N + j*N + N-2;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                }
            }
        }
    }
    dst = Eigen::SparseMatrix<float>(N*N*dims*fields,N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(D*D) * dst;
}


JFS_INLINE void fluid2D::div(Eigen::SparseMatrix<float> &dst, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*2*2);

    int dims = 2;

    for (int field=0; field < fields; field++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int iMat = j*N + i;
                int jMat = iMat;

                int dim = 0;
                
                //x+h,y
                jMat = field*dims*N*N + dim*N*N + j*N + (i+1);
                if ((i+1) >= N)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + dim*N*N + j*N + 1;  
                            tripletList.push_back(T(iMat,jMat,1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,1.f));
                
                //x-h,y
                jMat = field*dims*N*N + dim*N*N + j*N + i-1;
                if ((i-1) < 0)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + dim*N*N + j*N + N-2;  
                            tripletList.push_back(T(iMat,jMat,-1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,-1.f));

                dim = 1;
                
                //x,y+h
                jMat = field*dims*N*N + dim*N*N + (j+1)*N + i;
                if ((j+1) >= N)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + dim*N*N + 1*N + i;  
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
    }
    dst = Eigen::SparseMatrix<float>(N*N*fields,N*N*2*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}


JFS_INLINE void fluid2D::grad(Eigen::SparseMatrix<float> &dst, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*2*2);

    int dims = 1;

    for (int field=0; field < fields; field++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int iMat = j*N + i;
                int jMat = iMat;

                int dim = 0;
                iMat = field*dims*N*N + dim*N*N + j*N + i;
                
                //x+h,y
                jMat = field*dims*N*N + j*N + (i+1);
                if ((i+1) >= N)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + j*N + 1;  
                            tripletList.push_back(T(iMat,jMat,1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,1.f));
                
                //x-h,y
                jMat = field*dims*N*N + j*N + i-1;
                if ((i-1) < 0)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + j*N + N-2;  
                            tripletList.push_back(T(iMat,jMat,-1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,-1.f));

                dim = 1;
                iMat = field*dims*N*N + dim*N*N + j*N + i;
                
                //x,y+h
                jMat = field*dims*N*N + (j+1)*N + i;
                if ((j+1) >= N)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + 1*N + i;  
                            tripletList.push_back(T(iMat,jMat,1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,1.f));
                
                //x,y-h
                jMat = field*dims*N*N + (j-1)*N + i;
                if ((j-1) < 0)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + (N-2)*N + i;  
                            tripletList.push_back(T(iMat,jMat,-1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,-1.f));
            }
        }
    }

    dst = Eigen::SparseMatrix<float>(N*N*2*fields,N*N*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}


JFS_INLINE void fluid2D::calcLinInterp(Eigen::SparseMatrix<float> &dst, const Eigen::VectorXf &ij0, int dims, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*4*dims);
    
    for (int field=0; field < fields; field++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float i0 = ij0(0*N*N + N*j + i);
                float j0 = ij0(1*N*N + N*j + i);

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
                    iMat = field*dims*N*N + dim*N*N + j*N + i;
                    
                    jMat = field*dims*N*N + dim*N*N + j0_floor*N + i0_floor;
                    part = (i0_ceil - i0)*(j0_ceil - j0);
                    tripletList.push_back(T(iMat, jMat, std::abs(part)));
                    
                    jMat = field*dims*N*N + dim*N*N + j0_ceil*N + i0_floor;
                    if (j0_ceil < N)
                    {
                        part = (i0_ceil - i0)*(j0_floor - j0);
                        tripletList.push_back(T(iMat, jMat, std::abs(part)));
                    }
                    
                    jMat = field*dims*N*N + dim*N*N + j0_floor*N + i0_ceil;
                    if (i0_ceil < N)
                    {
                        part = (i0_floor - i0)*(j0_ceil - j0);
                        tripletList.push_back(T(iMat, jMat, std::abs(part)));
                    }
                    
                    jMat = field*dims*N*N + dim*N*N + j0_ceil*N + i0_ceil;
                    if (i0_ceil < N && j0_ceil < N)
                    {
                        part = (i0_floor - i0)*(j0_floor - j0);
                        tripletList.push_back(T(iMat, jMat, std::abs(part)));
                    }
                }
            }
        }        
    }

    dst = Eigen::SparseMatrix<float>(N*N*dims*fields,N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
}

} // namespace jfs