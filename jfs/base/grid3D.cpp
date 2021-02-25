#include <jfs/base/grid3D.h>
#include <iostream>

namespace jfs {

template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::satisfyBC(Vector_ &u)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

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

template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::Laplace(SparseMatrix_ &dst, unsigned int dims, unsigned int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    BOUND_TYPE tmp = BOUND;
    BOUND = ZERO;

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
    dst = SparseMatrix_(N*N*N*dims*fields,N*N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(D*D) * dst;

    BOUND = tmp;
}

template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::div(SparseMatrix_ &dst, unsigned int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

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
    dst = SparseMatrix_(N*N*N*fields,N*N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}

template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::grad(SparseMatrix_ &dst, unsigned int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

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

    dst = SparseMatrix_(N*N*N*3*fields,N*N*N*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}

template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::backstream(Vector_ &dst, const Vector_ &src, const Vector_ &ufield, float dt, FIELD_TYPE ftype, int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    int dims;
    switch (ftype)
    {
    case SCALAR_FIELD:
        dims = 1;
        break;

    case VECTOR_FIELD:
        dims = 3;
        break;
    }

    #pragma omp parallel
    #pragma omp for
    for (int index = 0; index < N*N*N; index++)
    {
        int k = std::floor(index/(N*N));
        int j = std::floor((index-N*N*k)/N);
        int i = index - N*N*k - N*j;
        Vector_ X(3);
        X(0) = D*(i + .5);
        X(1) = D*(j + .5);
        X(2) = D*(k + .5);

        X = this->sourceTrace(X, ufield, 3, -dt);

        Vector_ interp_indices = (X.array())/D - .5;

        Vector_ interp_quant = calcLinInterp(interp_indices, src, dims, fields);

        Eigen::VectorXi insert_indices(3);
        insert_indices(0) = i;
        insert_indices(1) = j;
        insert_indices(2) = k;

        insertIntoField(insert_indices, interp_quant, dst, dims, fields);
    }
}

template<int StorageOrder>
JFS_INLINE typename grid3D<StorageOrder>::Vector_ grid3D<StorageOrder>::
indexField(Eigen::VectorXi indices, const Vector_ &src, int dims, int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    int i = indices(0);
    int j = indices(1);
    int k = indices(2);

    Vector_ indexed_quant(dims*fields);

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            indexed_quant(dims*f + d) = src(N*N*N*dims*f + N*N*N*d + N*N*k + N*j + i);
        }
    }

    return indexed_quant;
}

template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::insertIntoField(Eigen::VectorXi indices, Vector_ q, Vector_ &dst, int dims, int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

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


template<int StorageOrder>
JFS_INLINE typename grid3D<StorageOrder>::Vector_ grid3D<StorageOrder>::
calcLinInterp(Vector_ interp_indices, const Vector_ &src, int dims, unsigned int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    Vector_ interp_quant(dims*fields);

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


template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::interpolateForce(const std::vector<Force> forces, SparseVector_ &dst)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;
    
    for (int f=0; f < forces.size(); f++)
    {
        const Force &force = forces[f];
        if (force.x>L || force.y>L) continue;
        
        float i = force.x/D;
        float j = force.y/D;
        float k = force.z/D;

        int i0 = std::floor(i);
        int j0 = std::floor(j);
        int k0 = std::floor(k);

        int dims = 3;
        float fArr[3] = {force.Fx, force.Fy, force.Fz};
        for (int dim=0; dim < dims; dim++)
        {
            dst.insert(N*N*N*dim + N*N*k0 + N*j0 + i0) += fArr[dim]*std::abs((j0+1 - j)*(i0+1 - i)*(k0+1 - i));
            if (i0 < (N-1))
                dst.insert(N*N*N*dim + N*N*k0 + N*j0 + (i0+1)) += fArr[dim]*std::abs((j0+1 - j)*(i0 - i)*(k0+1 - i));
            if (j0 < (N-1))
                dst.insert(N*N*N*dim + N*N*k0 + N*(j0+1) + i0) += fArr[dim]*std::abs((j0 - j)*(i0+1 - i)*(k0+1 - i));
            if (k0 < (N-1))
                dst.insert(N*N*N*dim + N*N*(k0+1) + N*j0 + i0) += fArr[dim]*std::abs((j0+1 - j)*(i0+1 - i)*(k0 - i));
            if (i0 < (N-1) && j0 < (N-1))
                dst.insert(N*N*N*dim + N*N*k0 + N*(j0+1) + (i0+1)) += fArr[dim]*std::abs((j0 - j)*(i0 - i)*(k0+1 - i));
            if (i0 < (N-1) && k0 < (N-1))
                dst.insert(N*N*N*dim + N*N*(k0+1) + N*j0 + (i0+1)) += fArr[dim]*std::abs((j0+1 - j)*(i0 - i)*(k0 - i));
            if (j0 < (N-1) && k0 < (N-1))
                dst.insert(N*N*N*dim + N*N*(k0+1) + N*(j0+1) + i0) += fArr[dim]*std::abs((j0 - j)*(i0+1 - i)*(k0 - i));
            if (i0 < (N-1) && j0 < (N-1) && k0 < (N-1))
                dst.insert(N*N*N*dim + N*N*(k0+1) + N*(j0+1) + (i0+1)) += fArr[dim]*std::abs((j0 - j)*(i0 - i)*(k0 - i));
        }
    }
}


template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::interpolateSource(const std::vector<Source> sources, SparseVector_ &dst)
{
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    for (int f=0; f < sources.size(); f++)
    {
        const Source &source = sources[f];
        if (source.x>L || source.y>L) continue;
        
        float i = source.x/D;
        float j = source.y/D;
        float k = source.z/D;

        int i0 = std::floor(i);
        int j0 = std::floor(j);
        int k0 = std::floor(k);

        for (int c=0; c < 3; c++)
        {
            float cval = {source.color(c) * source.strength};

            dst.insert(N*N*N*c + N*N*k0 + N*j0 + i0) += cval*std::abs((j0+1 - j)*(i0+1 - i)*(k0+1 - i));
            if (i0 < (N-1))
                dst.insert(N*N*N*c + N*N*k0 + N*j0 + (i0+1)) += cval*std::abs((j0+1 - j)*(i0 - i)*(k0+1 - i));
            if (j0 < (N-1))
                dst.insert(N*N*N*c + N*N*k0 + N*(j0+1) + i0) += cval*std::abs((j0 - j)*(i0+1 - i)*(k0+1 - i));
            if (k0 < (N-1))
                dst.insert(N*N*N*c + N*N*(k0+1) + N*j0 + i0) += cval*std::abs((j0+1 - j)*(i0+1 - i)*(k0 - i));
            if (i0 < (N-1) && j0 < (N-1))
                dst.insert(N*N*N*c + N*N*k0 + N*(j0+1) + (i0+1)) += cval*std::abs((j0 - j)*(i0 - i)*(k0+1 - i));
            if (i0 < (N-1) && k0 < (N-1))
                dst.insert(N*N*N*c + N*N*(k0+1) + N*j0 + (i0+1)) += cval*std::abs((j0+1 - j)*(i0 - i)*(k0 - i));
            if (j0 < (N-1) && k0 < (N-1))
                dst.insert(N*N*N*c + N*N*(k0+1) + N*(j0+1) + i0) += cval*std::abs((j0 - j)*(i0+1 - i)*(k0 - i));
            if (i0 < (N-1) && j0 < (N-1) && k0 < (N-1))
                dst.insert(N*N*N*c + N*N*(k0+1) + N*(j0+1) + (i0+1)) += cval*std::abs((j0 - j)*(i0 - i)*(k0 - i));
        }
    }
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class grid3D<Eigen::ColMajor>;
template class grid3D<Eigen::RowMajor>;
#endif

} // namespace jfs

