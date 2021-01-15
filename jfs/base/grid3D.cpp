#include <jfs/base/grid3D.h>
#include <iostream>

namespace jfs {

JFS_INLINE void grid3D::setXGrid()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                int dim = 0;
                X(dim*N*N*N + k*N*N + j*N + i) = D*(i+.5);
                
                dim = 1;
                X(dim*N*N*N + k*N*N + j*N + i) = D*(j+.5);
                
                dim = 2;
                X(dim*N*N*N + k*N*N + j*N + i) = D*(k+.5);
            }
        }
    }
}

JFS_INLINE void grid3D::initializeGrid(unsigned int N, float L, BOUND_TYPE BOUND, float dt)
{

    initializeGridProperties(N, L, BOUND, dt);

    X.resize(N*N*N*3);
    setXGrid();

    Laplace(LAPLACE, 1);
    Laplace(VEC_LAPLACE, 3);
    div(DIV);
    grad(GRAD);

    ij0.resize(N*N*N*3);
    linInterp.resize(N*N*N,N*N*N);
    linInterpVec.resize(N*N*N*3,N*N*N*3);
}


JFS_INLINE void grid3D::satisfyBC(Eigen::VectorXf &u)
{
    int i,j,k;
    if (BOUND == PERIODIC)
    for (int idx1=0; idx1 < N; idx1++)
        for (int idx2=0; idx2 < N; idx2++)
        {
            // top
            i = idx1;
            j = N-1;
            k = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = u(N*N*N*0 + N*N*k + N*(j-(N-1)) + i);
            u(N*N*N*1 + N*N*k + N*j + i) = u(N*N*N*1 + N*N*k + N*(j-(N-1)) + i);
            u(N*N*N*2 + N*N*k + N*j + i) = u(N*N*N*2 + N*N*k + N*(j-(N-1)) + i);

            // bottom
            i = idx1;
            j = 0;
            k = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = u(N*N*N*0 + N*N*k + N*(j+(N-1)) + i);
            u(N*N*N*1 + N*N*k + N*j + i) = u(N*N*N*1 + N*N*k + N*(j+(N-1)) + i);
            u(N*N*N*2 + N*N*k + N*j + i) = u(N*N*N*2 + N*N*k + N*(j+(N-1)) + i);

            // left
            j = idx1;
            i = 0;
            k = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = u(N*N*N*0 + N*N*k + N*j + (i+(N-1)));
            u(N*N*N*1 + N*N*k + N*j + i) = u(N*N*N*1 + N*N*k + N*j + (i+(N-1)));
            u(N*N*N*2 + N*N*k + N*j + i) = u(N*N*N*2 + N*N*k + N*j + (i+(N-1)));

            // right
            j = idx1;
            i = N-1;
            k = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = u(N*N*N*0 + N*N*k + N*j + (i-(N-1)));
            u(N*N*N*1 + N*N*k + N*j + i) = u(N*N*N*1 + N*N*k + N*j + (i-(N-1)));
            u(N*N*N*2 + N*N*k + N*j + i) = u(N*N*N*2 + N*N*k + N*j + (i-(N-1)));

            // back
            k = 0;
            i = idx1;
            j = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = u(N*N*N*0 + N*N*(k+(N-1)) + N*j + i);
            u(N*N*N*1 + N*N*k + N*j + i) = u(N*N*N*1 + N*N*(k+(N-1)) + N*j + i);
            u(N*N*N*2 + N*N*k + N*j + i) = u(N*N*N*2 + N*N*(k+(N-1)) + N*j + i);

            // front
            k = N-1;
            i = idx1;
            j = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = u(N*N*N*0 + N*N*(k-(N-1)) + N*j + i);
            u(N*N*N*1 + N*N*k + N*j + i) = u(N*N*N*1 + N*N*(k-(N-1)) + N*j + i);
            u(N*N*N*2 + N*N*k + N*j + i) = u(N*N*N*2 + N*N*(k-(N-1)) + N*j + i);
        }

    else if (BOUND == ZERO)
    for (int idx1=0; idx1 < N; idx1++)
        for (int idx2=0; idx2 < N; idx2++)
        {
            // top
            i = idx1;
            j = N-1;
            k = idx2;
            u(N*N*N*1 + N*N*k + N*j + i) = 0;

            // bottom
            i = idx1;
            j = 0;
            k = idx2;
            u(N*N*N*1 + N*N*k + N*j + i) = 0;

            // left
            j = idx1;
            i = 0;
            k = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = 0;

            // right
            j = idx1;
            i = N-1;
            k = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = 0;

            // back
            k = 0;
            i = idx1;
            j = idx2;
            u(N*N*N*2 + N*N*k + N*j + i) = 0;

            // front
            k = N-1;
            i = idx1;
            j = idx2;
            u(N*N*N*2 + N*N*k + N*j + i) = 0;
        }
}


JFS_INLINE void grid3D::Laplace(SparseMatrix &dst, unsigned int dims, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*N*5*dims*fields);

    for (int field = 0; field < fields; field++)
    {
        for (int dim = 0; dim < dims; dim++)
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    for (int k = 0; k < N; k++)
                    {
                        int iMat = field*dims*N*N*N + dim*N*N*N + k*N*N + j*N + i;
                        int jMat = iMat;
                        
                        // x,y,z
                        tripletList.push_back(T(iMat,jMat,-6.f));
                        
                        //x,y+h,z
                        jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + (j+1)*N + i;
                        if ((j+1) >= N)
                            switch (BOUND)
                            {
                                case ZERO:
                                    break;

                                case PERIODIC:
                                    jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + 1*N + i;  
                                    tripletList.push_back(T(iMat,jMat,1.f)); 
                            }
                        else
                            tripletList.push_back(T(iMat,jMat,1.f));
                        
                        //x,y-h,z
                        jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + (j-1)*N + i;
                        if ((j-1) < 0)
                            switch (BOUND)
                            {
                                case ZERO:
                                    break;

                                case PERIODIC:
                                    jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + (N-2)*N + i;  
                                    tripletList.push_back(T(iMat,jMat,1.f)); 
                            }
                        else
                            tripletList.push_back(T(iMat,jMat,1.f));
                        
                        //x+h,y,z
                        jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + j*N + (i+1);
                        if ((i+1) >= N)
                            switch (BOUND)
                            {
                                case ZERO:
                                    break;

                                case PERIODIC:
                                    jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + j*N + 1;  
                                    tripletList.push_back(T(iMat,jMat,1.f)); 
                            }
                        else
                            tripletList.push_back(T(iMat,jMat,1.f));
                        
                        //x-h,y,z
                        jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + j*N + (i-1);
                        if ((i-1) < 0)
                            switch (BOUND)
                            {
                                case ZERO:
                                    break;

                                case PERIODIC:
                                    jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + j*N + (N-2);  
                                    tripletList.push_back(T(iMat,jMat,1.f)); 
                            }
                        else
                            tripletList.push_back(T(iMat,jMat,1.f));
                        
                        //x,y,z+h
                        jMat = field*dims*N*N*N + dim*N*N*N + (k+1)*N*N + j*N + i;
                        if ((k+1) >= N)
                            switch (BOUND)
                            {
                                case ZERO:
                                    break;

                                case PERIODIC:
                                    jMat = field*dims*N*N*N + dim*N*N*N + 1*N*N + j*N + i;  
                                    tripletList.push_back(T(iMat,jMat,1.f)); 
                            }
                        else
                            tripletList.push_back(T(iMat,jMat,1.f));
                        
                        //x,y,z-h
                        jMat = field*dims*N*N*N + dim*N*N*N + (k-1)*N*N + j*N + i;
                        if ((k-1) < 0)
                            switch (BOUND)
                            {
                                case ZERO:
                                    break;

                                case PERIODIC:
                                    jMat = field*dims*N*N*N + dim*N*N*N + (N-2)*N*N + j*N + i;  
                                    tripletList.push_back(T(iMat,jMat,1.f)); 
                            }
                        else
                            tripletList.push_back(T(iMat,jMat,1.f));
                    }
                }
            }
        }
    }
    dst = SparseMatrix(N*N*N*dims*fields,N*N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(D*D) * dst;
}


JFS_INLINE void grid3D::div(SparseMatrix &dst, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*N*3*2*fields);

    int dims = 3;

    for (int field=0; field < fields; field++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int k = 0; k < N; k++)
                {
                    int iMat = field*N*N*N + k*N*N + j*N + i;
                    int jMat = iMat;

                    int dim = 0;
                    
                    //x+h,y,z
                    jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + j*N + (i+1);
                    if ((i+1) >= N)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + j*N + 1;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x-h,y,z
                    jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + j*N + (i-1);
                    if ((i-1) < 0)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + j*N + (N-2);  
                                tripletList.push_back(T(iMat,jMat,-1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,-1.f));

                    dim = 1;
                    
                    //x,y+h,z
                    jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + (j+1)*N + i;
                    if ((j+1) >= N)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + 1*N + i;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x,y-h,z
                    jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + (j-1)*N + i;
                    if ((j-1) < 0)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + dim*N*N*N + k*N*N + (N-2)*N + i; 
                                tripletList.push_back(T(iMat,jMat,-1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,-1.f));

                    dim = 2;
                    
                    //x,y,z+h
                    jMat = field*dims*N*N*N + dim*N*N*N + (k+1)*N*N + j*N + i;
                    if ((k+1) >= N)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + dim*N*N*N + 1*N*N + j*N + i;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x,y,z-h
                    jMat = field*dims*N*N*N + dim*N*N*N + (k-1)*N*N + j*N + i;
                    if ((k-1) < 0)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + dim*N*N*N + (N-2)*N*N + j*N + i;  
                                tripletList.push_back(T(iMat,jMat,-1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,-1.f));
                }
            }
        }
    }
    dst = SparseMatrix(N*N*N*fields,N*N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}


JFS_INLINE void grid3D::grad(SparseMatrix &dst, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(fields*N*N*N*3*2);

    int dims = 1;

    for (int field=0; field < fields; field++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int k = 0; k < N; k++)
                {
                    int iMat = field*N*N*N + k*N*N + j*N + i;
                    int jMat = iMat;

                    int dim = 0;
                    iMat = field*3*N*N*N + dim*N*N*N + k*N*N + j*N + i;
                    
                    //x+h,y,z
                    jMat = field*dims*N*N*N + k*N*N + j*N + (i+1);
                    if ((i+1) >= N)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + k*N*N + j*N + 1;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x-h,y,z
                    jMat = field*dims*N*N*N + k*N*N + j*N + (i-1);
                    if ((i-1) < 0)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + k*N*N + j*N + (N-2);
                                tripletList.push_back(T(iMat,jMat,-1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,-1.f));

                    dim = 1;
                    iMat = field*3*N*N*N + dim*N*N*N + k*N*N + j*N + i;
                    
                    //x,y+h,z
                    jMat = field*dims*N*N*N + k*N*N + (j+1)*N + i;
                    if ((j+1) >= N)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + k*N*N + 1*N + i; 
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x,y-h,z
                    jMat = field*dims*N*N*N + k*N*N + (j-1)*N + i;
                    if ((j-1) < 0)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + k*N*N + (N-2)*N + i; 
                                tripletList.push_back(T(iMat,jMat,-1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,-1.f));

                    dim = 2;
                    iMat = field*3*N*N*N + dim*N*N*N + k*N*N + j*N + i;
                    
                    //x,y,z+h
                    jMat = field*dims*N*N*N + (k+1)*N*N + j*N + i;
                    if ((k+1) >= N)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + 1*N*N + j*N + i;
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x,y,z-h
                    jMat = field*dims*N*N*N + (k-1)*N*N + j*N + i;
                    if ((k-1) < 0)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N*N + (N-2)*N*N + j*N + i;
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                }
            }
        }
    }

    dst = SparseMatrix(N*N*N*3*fields,N*N*N*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}


JFS_INLINE void grid3D::calcLinInterp(SparseMatrix &dst, const Eigen::VectorXf &ij0, int dims, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*N*dims*fields*8);
    
    for (int field=0; field < fields; field++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int k = 0; k < N; k++)
                {
                    float i0 = ij0(0*N*N*N + N*N*k + N*j + i);
                    float j0 = ij0(1*N*N*N + N*N*k + N*j + i);
                    float k0 = ij0(2*N*N*N + N*N*k + N*j + i);

                    switch (BOUND)
                    {
                        case ZERO:
                            i0 = (i0 < 0) ? 0:i0;
                            i0 = (i0 > (N-1)) ? (N-1):i0;
                            j0 = (j0 < 0) ? 0:j0;
                            j0 = (j0 > (N-1)) ? (N-1):j0;
                            k0 = (j0 < 0) ? 0:j0;
                            k0 = (j0 > (N-1)) ? (N-1):j0;
                            break;
                        
                        case PERIODIC:
                            while (i0 < 0 || i0 > N-1 || j0 < 0 || j0 > N-1)
                            {
                                i0 = (i0 < 0) ? (N-1+i0):i0;
                                i0 = (i0 > (N-1)) ? (i0 - (N-1)):i0;
                                j0 = (j0 < 0) ? (N-1+j0):j0;
                                j0 = (j0 > (N-1)) ? (j0 - (N-1)):j0;
                                k0 = (k0 < 0) ? (N-1+k0):k0;
                                k0 = (k0 > (N-1)) ? (k0 - (N-1)):k0;
                            }
                            break;
                    }

                    int i0_floor = (int) i0;
                    int j0_floor = (int) j0;
                    int k0_floor = (int) k0;
                    int i0_ceil = i0_floor + 1;
                    int j0_ceil = j0_floor + 1;
                    int k0_ceil = k0_floor + 1;
                    int iMat, jMat, iref, jref, kref;
                    float part;

                    for (int dim = 0; dim < dims; dim++)
                    {
                        iMat = field*dims*N*N*N + dim*N*N*N + k*N*N + j*N + i;


                        // 1
                        jMat = field*dims*N*N*N + dim*N*N*N + k0_floor*N*N + j0_floor*N + i0_floor;
                        part = (i0_ceil - i0)*(j0_ceil - j0)*(k0_ceil - k0);
                        tripletList.push_back(T(iMat, jMat, std::abs(part)));
                        // if (jMat > N*N*dims*fields-1 || jMat < 0)
                        // {
                        //     printf("NO!\n");
                        // }
                        
                        // 2-4
                        jMat = field*dims*N*N*N + dim*N*N*N + k0_floor*N*N + j0_floor*N + i0_ceil;
                        if (i0_ceil < N)
                        {
                            part = (i0_floor - i0)*(j0_ceil - j0)*(k0_ceil - k0);
                            tripletList.push_back(T(iMat, jMat, std::abs(part)));
                        // if (jMat > N*N*dims*fields-1 || jMat < 0)
                        // {
                        //     printf("NO!\n");
                        // }
                        }
                        
                        jMat = field*dims*N*N*N + dim*N*N*N + k0_floor*N*N + j0_ceil*N + i0_floor;
                        if (j0_ceil < N)
                        {
                            part = (i0_ceil- i0)*(j0_floor - j0)*(k0_ceil - k0);
                            tripletList.push_back(T(iMat, jMat, std::abs(part)));
                        // if (jMat > N*N*dims*fields-1 || jMat < 0)
                        // {
                        //     printf("NO!\n");
                        // }
                        }
                        
                        jMat = field*dims*N*N*N + dim*N*N*N + k0_ceil*N*N + j0_floor*N + i0_floor;
                        if (k0_ceil < N)
                        {
                            part = (i0_ceil - i0)*(j0_ceil - j0)*(k0_floor - k0);
                            tripletList.push_back(T(iMat, jMat, std::abs(part)));
                        // if (jMat > N*N*dims*fields-1 || jMat < 0)
                        // {
                        //     printf("NO!\n");
                        // }
                        }                    
                        
                        // 5-7
                        jMat = field*dims*N*N*N + dim*N*N*N + k0_floor*N*N + j0_ceil*N + i0_ceil;
                        if (i0_ceil < N && j0_ceil < N)
                        {
                            part = (i0_floor - i0)*(j0_floor - j0)*(k0_ceil - k0);
                            tripletList.push_back(T(iMat, jMat, std::abs(part)));
                        // if (jMat > N*N*dims*fields-1 || jMat < 0)
                        // {
                        //     printf("NO!\n");
                        // }
                        }

                        jMat = field*dims*N*N*N + dim*N*N*N + k0_ceil*N*N + j0_floor*N + i0_ceil;
                        if (i0_ceil < N && k0_ceil < N)
                        {
                            part = (i0_floor - i0)*(j0_ceil - j0)*(k0_floor - k0);
                            tripletList.push_back(T(iMat, jMat, std::abs(part)));
                        // if (jMat > N*N*dims*fields-1 || jMat < 0)
                        // {
                        //     printf("NO!\n");
                        // }
                        }
                        
                        jMat = field*dims*N*N*N + dim*N*N*N + k0_ceil*N*N + j0_ceil*N + i0_floor;
                        if (j0_ceil < N && k0_ceil < N)
                        {
                            part = (i0_ceil - i0)*(j0_floor - j0)*(k0_floor - k0);
                            tripletList.push_back(T(iMat, jMat, std::abs(part)));
                        // if (jMat > N*N*dims*fields-1 || jMat < 0)
                        // {
                        //     printf("NO!\n");
                        // }
                        }
                        
                        // 8
                        jMat = field*dims*N*N*N + dim*N*N*N + k0_ceil*N*N + j0_ceil*N + i0_ceil;
                        if (i0_ceil < N && j0_ceil < N && k0_ceil < N)
                        {
                            part = (i0_floor - i0)*(j0_floor - j0)*(k0_floor - k0);
                            tripletList.push_back(T(iMat, jMat, std::abs(part)));
                        // if (jMat > N*N*dims*fields-1 || jMat < 0)
                        // {
                        //     printf("NO!\n");
                        // }
                        }
                    }
                }
            }
        }        
    }

    dst = SparseMatrix(N*N*N*fields*dims,N*N*N*fields*dims);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
}

} // namespace jfs

