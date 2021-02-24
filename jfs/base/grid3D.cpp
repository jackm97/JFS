#include <jfs/base/grid3D.h>
#include <iostream>

namespace jfs {

JFS_INLINE void grid3D::initializeGrid(unsigned int N, float L, BOUND_TYPE BOUND, float dt)
{

    initializeGridProperties(N, L, BOUND, dt);

    Laplace(LAPLACE, 1);
    Laplace(VEC_LAPLACE, 3);
    div(DIV);
    grad(GRAD);
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
            i = 0;
            j = idx1;
            k = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = u(N*N*N*0 + N*N*k + N*j + (i+(N-1)));
            u(N*N*N*1 + N*N*k + N*j + i) = u(N*N*N*1 + N*N*k + N*j + (i+(N-1)));
            u(N*N*N*2 + N*N*k + N*j + i) = u(N*N*N*2 + N*N*k + N*j + (i+(N-1)));

            // right
            i = N-1;
            j = idx1;
            k = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = u(N*N*N*0 + N*N*k + N*j + (i-(N-1)));
            u(N*N*N*1 + N*N*k + N*j + i) = u(N*N*N*1 + N*N*k + N*j + (i-(N-1)));
            u(N*N*N*2 + N*N*k + N*j + i) = u(N*N*N*2 + N*N*k + N*j + (i-(N-1)));

            // back
            i = idx1;
            j = idx2;
            k = 0;
            u(N*N*N*0 + N*N*k + N*j + i) = u(N*N*N*0 + N*N*(k+(N-1)) + N*j + i);
            u(N*N*N*1 + N*N*k + N*j + i) = u(N*N*N*1 + N*N*(k+(N-1)) + N*j + i);
            u(N*N*N*2 + N*N*k + N*j + i) = u(N*N*N*2 + N*N*(k+(N-1)) + N*j + i);

            // front
            i = idx1;
            j = idx2;
            k = N-1;
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
            i = 0;
            j = idx1;
            k = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = 0;

            // right
            i = N-1;
            j = idx1;
            k = idx2;
            u(N*N*N*0 + N*N*k + N*j + i) = 0;

            // back
            i = idx1;
            j = idx2;
            k = 0;
            u(N*N*N*2 + N*N*k + N*j + i) = 0;

            // front
            i = idx1;
            j = idx2;
            k = N-1;
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
                                tripletList.push_back(T(iMat,jMat,-1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,-1.f));
                }
            }
        }
    }

    dst = SparseMatrix(N*N*N*3*fields,N*N*N*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}

JFS_INLINE void grid3D::backstream(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &ufield, float dt, int dims, int fields)
{
    for (int index = 0; index < N*N*N; index++)
    {
        int k = std::floor(index/(N*N));
        int j = std::floor((index-N*N*k)/N);
        int i = index - N*N*k - N*j;
        Eigen::VectorXf X(3);
        X(0) = D*(i + .5);
        X(1) = D*(j + .5);
        X(2) = D*(k + .5);

        X = sourceTrace(X, ufield, 3, -dt);

        Eigen::VectorXf interp_indices = (X.array())/D - .5;

        Eigen::VectorXf interp_quant = calcLinInterp(interp_indices, src, dims, fields);

        Eigen::VectorXi insert_indices(3);
        insert_indices(0) = i;
        insert_indices(1) = j;
        insert_indices(2) = k;

        insertIntoField(insert_indices, interp_quant, dst, dims, fields);
    }
}

JFS_INLINE Eigen::VectorXf grid3D::indexField(Eigen::VectorXi indices, const Eigen::VectorXf &src, int dims, int fields)
{
    int i = indices(0);
    int j = indices(1);
    int k = indices(2);

    Eigen::VectorXf indexed_quant(dims*fields);

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            indexed_quant(dims*f + d) = src(N*N*N*dims*f + N*N*N*d + N*N*k + N*j + i);
        }
    }

    return indexed_quant;
}

JFS_INLINE void grid3D::insertIntoField(Eigen::VectorXi indices, Eigen::VectorXf q, Eigen::VectorXf &dst, int dims, int fields)
{
    int i = indices(0);
    int j = indices(1);
    int k = indices(2);

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            dst(N*N*N*dims*f + N*N*N*d + N*N*k + N*j + i) = q(dims*f + d);
        }
    }
}


JFS_INLINE Eigen::VectorXf grid3D::calcLinInterp(Eigen::VectorXf interp_indices, const Eigen::VectorXf &src, int dims, unsigned int fields)
{
    Eigen::VectorXf interp_quant(dims*fields);

    float i0 = interp_indices(0);
    float j0 = interp_indices(1);
    float k0 = interp_indices(2);

    switch (BOUND)
    {
        case ZERO:
            i0 = (i0 < 0) ? 0:i0;
            i0 = (i0 > (N-1)) ? (N-1):i0;
            j0 = (j0 < 0) ? 0:j0;
            j0 = (j0 > (N-1)) ? (N-1):j0;
            k0 = (k0 < 0) ? 0:k0;
            k0 = (k0 > (N-1)) ? (N-1):k0;
            break;
        
        case PERIODIC:
            while (i0 < 0 || i0 > N-1 || j0 < 0 || j0 > N-1 || k0 < 0 || k0 > N-1)
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
    float part;

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {


            // 1
            part = (i0_ceil - i0)*(j0_ceil - j0)*(k0_ceil - k0);
            part = std::abs(part);
            float q1 = part * src(f*dims*N*N*N + d*N*N*N + k0_floor*N*N + j0_floor*N + i0_floor);
            
            // 2-4
            float q2;
            if (i0_ceil < N)
            {
                part = (i0_floor - i0)*(j0_ceil - j0)*(k0_ceil - k0);
                part = std::abs(part);
                q2 = part * src(f*dims*N*N*N + d*N*N*N + k0_floor*N*N + j0_floor*N + i0_ceil);
            }
            else
                q2 = 0;

            float q3;
            if (j0_ceil < N)
            {
                part = (i0_ceil - i0)*(j0_floor - j0)*(k0_ceil - k0);
                part = std::abs(part);
                q3 = part * src(f*dims*N*N*N + d*N*N*N + k0_floor*N*N + j0_ceil*N + i0_floor);
            }
            else
                q3 = 0; 

            float q4;
            if (k0_ceil < N)
            {
                part = (i0_ceil - i0)*(j0_ceil - j0)*(k0_floor - k0);
                part = std::abs(part);
                q4 = part * src(f*dims*N*N*N + d*N*N*N + k0_ceil*N*N + j0_floor*N + i0_floor);
            }
            else
                q4 = 0;                   
            
            // 5-7
            float q5;
            if (i0_ceil < N && j0_ceil < N)
            {
                part = (i0_floor - i0)*(j0_floor - j0)*(k0_ceil - k0);
                part = std::abs(part);
                q5 = part * src(f*dims*N*N*N + d*N*N*N + k0_floor*N*N + j0_ceil*N + i0_ceil);
            }
            else
                q5 = 0; 

            float q6;
            if (i0_ceil < N && k0_ceil < N)
            {
                part = (i0_floor - i0)*(j0_ceil - j0)*(k0_floor - k0);
                part = std::abs(part);
                q6 = part * src(f*dims*N*N*N + d*N*N*N + k0_ceil*N*N + j0_floor*N + i0_ceil);
            }
            else
                q6 = 0; 

            float q7;
            if (j0_ceil < N && k0_ceil < N)
            {
                part = (i0_ceil - i0)*(j0_floor - j0)*(k0_floor - k0);
                part = std::abs(part);
                q7 = part * src(f*dims*N*N*N + d*N*N*N + k0_ceil*N*N + j0_ceil*N + i0_floor);
            }
            else
                q7 = 0; 
            
            // 8
            float q8;
            if (i0_ceil < N && j0_ceil < N && k0_ceil < N)
            {
                part = (i0_floor - i0)*(j0_floor - j0)*(k0_floor - k0);
                part = std::abs(part);
                q8 = part * src(f*dims*N*N*N + d*N*N*N + k0_ceil*N*N + j0_ceil*N + i0_ceil);
            }
            else
                q8 = 0; 

            interp_quant(dims*f + d) = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8;
        }
    }

    return interp_quant;
}

} // namespace jfs

