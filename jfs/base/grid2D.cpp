#include <jfs/base/grid2D.h>

namespace jfs {


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::satisfyBC(Vector_ &u)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    int i,j;
    if (BOUND == PERIODIC)
    for (int idx=0; idx < N; idx++)
    {
        // top
        i = idx;
        j = N-1;
        u(N*N*0 + N*j + i) = u(N*N*0 + N*(j-(N-1)) + i);
        u(N*N*1 + N*j + i) = u(N*N*1 + N*(j-(N-1)) + i);

        // bottom
        i = idx;
        j = 0;
        u(N*N*0 + N*j + i) = u(N*N*0 + N*(j+(N-1)) + i);
        u(N*N*1 + N*j + i) = u(N*N*1 + N*(j+(N-1)) + i);

        // left
        j = idx;
        i = N-1;
        u(N*N*0 + N*j + i) = u(N*N*0 + N*j + (i-(N-1)));
        u(N*N*1 + N*j + i) = u(N*N*1 + N*j + (i-(N-1)));

        // right
        j = idx;
        i = 0;
        u(N*N*0 + N*j + i) = u(N*N*0 + N*j + (i+(N-1)));
        u(N*N*1 + N*j + i) = u(N*N*1 + N*j + (i+(N-1)));
    }

    else if (BOUND == ZERO)
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


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::Laplace(SparseMatrix_ &dst, unsigned int dims, unsigned int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*5*dims);

    BOUND_TYPE tmp = BOUND;
    BOUND = ZERO;

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
    dst = SparseMatrix_(N*N*dims*fields,N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(D*D) * dst;

    BOUND = tmp;
}


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::div(SparseMatrix_ &dst, unsigned int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

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
    dst = SparseMatrix_(N*N*fields,N*N*2*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::grad(SparseMatrix_ &dst, unsigned int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

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
                iMat = field*2*N*N + dim*N*N + j*N + i;
                
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
                iMat = field*2*N*N + dim*N*N + j*N + i;
                
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

    dst = SparseMatrix_(N*N*2*fields,N*N*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}

template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::backstream(Vector_ &dst, const Vector_ &src, const Vector_ &ufield, float dt, FIELD_TYPE ftype, int fields)
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
        dims = 2;
        break;
    }

    #pragma omp parallel
    #pragma omp for
    for (int index = 0; index < N*N; index++)
    {
        int j = std::floor(index/N);
        int i = index - N*j;
        Vector_ X(2);
        X(0) = D*(i + .5);
        X(1) = D*(j + .5);

        X = this->sourceTrace(X, ufield, 2, -dt);

        Vector_ interp_indices = (X.array())/D - .5;

        Vector_ interp_quant = calcLinInterp(interp_indices, src, dims, fields);

        Eigen::VectorXi insert_indices(2);
        insert_indices(0) = i;
        insert_indices(1) = j;

        insertIntoField(insert_indices, interp_quant, dst, dims, fields);
    }
}

template<int StorageOrder>
JFS_INLINE typename grid2D<StorageOrder>::Vector_ grid2D<StorageOrder>::indexField(Eigen::VectorXi indices, const Vector_ &src, int dims, int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    int i = indices(0);
    int j = indices(1);

    Vector_ indexed_quant(dims*fields);

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            indexed_quant(dims*f + d) = src(N*N*dims*f + N*N*d + N*j + i);
        }
    }

    return indexed_quant;
}

template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::insertIntoField(Eigen::VectorXi indices, Vector_ q, Vector_ &dst, int dims, int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    int i = indices(0);
    int j = indices(1);

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            dst(N*N*dims*f + N*N*d + N*j + i) = q(dims*f + d);
        }
    }
}


template<int StorageOrder>
JFS_INLINE typename grid2D<StorageOrder>::Vector_ grid2D<StorageOrder>::calcLinInterp(Vector_ interp_indices, const Vector_ &src, int dims, unsigned int fields)
{
    auto BOUND = this->BOUND;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    Vector_ interp_quant(dims*fields);

    float i0 = interp_indices(0);
    float j0 = interp_indices(1);

    switch (BOUND)
    {
        case ZERO:
            i0 = (i0 < 0) ? 0:i0;
            i0 = (i0 > (N-1)) ? (N-1):i0;
            j0 = (j0 < 0) ? 0:j0;
            j0 = (j0 > (N-1)) ? (N-1):j0;
            break;
        
        case PERIODIC:
            while (i0 < 0 || i0 > N-1 || j0 < 0 || j0 > N-1)
            {
                i0 = (i0 < 0) ? (N-1+i0):i0;
                i0 = (i0 > (N-1)) ? (i0 - (N-1)):i0;
                j0 = (j0 < 0) ? (N-1+j0):j0;
                j0 = (j0 > (N-1)) ? (j0 - (N-1)):j0;
            }
            break;
    }

    int i0_floor = (int) i0;
    int j0_floor = (int) j0;
    int i0_ceil = i0_floor + 1;
    int j0_ceil = j0_floor + 1;
    int iMat, jMat, iref, jref;
    float part;

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            part = (i0_ceil - i0)*(j0_ceil - j0);
            part = std::abs(part);
            float q1 = part * src(N*N*dims*f + N*N*d + N*j0_floor + i0_floor);

            float q2;
            if (j0_ceil < N)
            {
                part = (i0_ceil - i0)*(j0_floor - j0);
                part = std::abs(part);
                q2 = part * src(N*N*dims*f + N*N*d + N*j0_ceil + i0_floor); 
            }
            else
            {
                q2 = 0;
            }

            float q3;
            if (i0_ceil < N)
            {
                part = (i0_floor - i0)*(j0_ceil - j0);
                part = std::abs(part);
                q3 = part * src(N*N*dims*f + N*N*d + N*j0_floor + i0_ceil); 
            }
            else
            {
                q3 = 0;
            }

            float q4;
            if (i0_ceil < N && j0_ceil < N)
            {
                part = (i0_floor - i0)*(j0_floor - j0);
                part = std::abs(part);
                q4 = part * src(N*N*dims*f + N*N*d + N*j0_ceil + i0_ceil); 
            }
            else
            {
                q4 = 0;
            }

            interp_quant(dims*f + d) = q1 + q2 + q3 + q4;
        }
    }

    return interp_quant;
}


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::interpolateForce(const std::vector<Force> forces, SparseVector_ &dst)
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

        int i0 = std::floor(i);
        int j0 = std::floor(j);

        int dims = 2;
        float fArr[2] = {force.Fx, force.Fy};
        for (int dim=0; dim < dims; dim++)
        {
            dst.insert(N*N*dim + N*j0 + i0) += fArr[dim]*std::abs((j0+1 - j)*(i0+1 - i));
            if (i0 < (N-1))
                dst.insert(N*N*dim + N*j0 + (i0+1)) += fArr[dim]*std::abs((j0+1 - j)*(i0 - i));
            if (j0 < (N-1))
                dst.insert(N*N*dim + N*(j0+1) + i0) += fArr[dim]*std::abs((j0 - j)*(i0+1 - i));
            if (i0 < (N-1) && j0 < (N-1))
                dst.insert(N*N*dim + N*(j0+1) + (i0+1)) += fArr[dim]*std::abs((j0 - j)*(i0 - i));
        }
    }
}


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::interpolateSource(const std::vector<Source> sources, SparseVector_ &dst)
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

        int i0 = std::floor(i);
        int j0 = std::floor(j);

        for (int c=0; c < 3; c++)
        {
            float cval = {source.color(c) * source.strength};
            dst.insert(c*N*N + N*j0 + i0) += cval*std::abs((j0+1 - j)*(i0+1 - i));
            if (i0 < (N-1))
                dst.insert(c*N*N + N*j0 + (i0+1)) += cval*std::abs((j0+1 - j)*(i0 - i));
            if (j0 < (N-1))
                dst.insert(c*N*N + N*(j0+1) + i0) += cval*std::abs((j0 - j)*(i0+1 - i));
            if (i0 < (N-1) && j0 < (N-1))            
                dst.insert(c*N*N + N*(j0+1) + (i0+1)) += cval*std::abs((j0 - j)*(i0 - i));
        }
    }
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class grid2D<Eigen::ColMajor>;
template class grid2D<Eigen::RowMajor>;
#endif

} // namespace jfs

