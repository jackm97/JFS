#include <jfs/base/grid2D.h>

namespace jfs {


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::satisfyBC(Vector_ &dst, FieldType ftype, int fields)
{
    auto btype = this->bound_type_;
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

    int i,j;
    if (btype == PERIODIC)
    for (int idx=0; idx < N; idx++)
    {
        for (int f = 0; f < fields; f++)
        {
            for (int d = 0; d < dims; d++)
            {
                // bottom
                i = idx;
                j = 0;
                dst(f*dims*N*N + d*N*N + N*j + i) = dst(f*dims*N*N + d*N*N + N*(N-1) + i);

                // left
                j = idx;
                i = 0;
                dst(f*dims*N*N + d*N*N + N*j + i) = dst(f*dims*N*N + d*N*N + N*j + (N-1));
            }
        }
    }

    else if (btype == ZERO)
    for (int idx=0; idx < N; idx++)
    {
        for (int f = 0; f < fields; f++)
        {
            for (int d = 0; d < dims; d++)
            {
                // top
                i = idx;
                j = N-1;
                if (ftype == SCALAR_FIELD || d == 1)
                    dst(f*dims*N*N + d*N*N + N*j + i) = 0;

                // bottom
                i = idx;
                j = 0;
                if (ftype == SCALAR_FIELD || d == 1)
                    dst(f*dims*N*N + d*N*N + N*j + i) = 0;

                // left
                j = idx;
                i = N-1;
                if (ftype == SCALAR_FIELD || d == 0)
                    dst(f*dims*N*N + d*N*N + N*j + i) = 0;

                // right
                j = idx;
                i = 0;
                if (ftype == SCALAR_FIELD || d == 0)
                    dst(f*dims*N*N + d*N*N + N*j + i) = 0;
            }
        }
    }
}


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::Laplace(SparseMatrix_ &dst, unsigned int dims, unsigned int fields)
{
    auto btype = this->bound_type_;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*5*dims);

    for (int idx = 0; idx < dims*fields*N*N; idx++)
    {
        int idx_tmp = idx;
        int f = idx_tmp / (dims * N * N);
        idx_tmp -= f * (dims * N * N);
        int d = idx_tmp / (N * N);
        idx_tmp -= d * (N * N);
        int j = idx_tmp / N;
        idx_tmp -= j * N;
        int i = idx_tmp;

        int iMat, jMat;

        iMat = f*dims*N*N + d*N*N + j*N + i;
        jMat = iMat;
        tripletList.push_back(T(iMat,jMat,-4.f));

        int i_tmp = i;
        for (int offset = -1; offset < 2; offset+=2)
        {
            i = i_tmp + offset;
            if ( (i == -1 || i == N) && btype == ZERO)
                continue;
            else if ( i == -1 )
                i = (N-2);
            else if ( i == N )
                i = 1;
            jMat = f*dims*N*N + d*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,1.f));
        }
        i = i_tmp;

        int j_tmp = j;
        for (int offset = -1; offset < 2; offset+=2)
        {
            j = j_tmp + offset;
            if ( (j == -1 || j == N) && btype == ZERO)
                continue;
            else if ( j == -1 )
                j = (N-2);
            else if ( j == N )
                j = 1;
            jMat = f*dims*N*N + d*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,1.f));
        }
        j = j_tmp;
    }

    dst = SparseMatrix_(N*N*dims*fields,N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(D*D) * dst;
}


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::div(SparseMatrix_ &dst, unsigned int fields)
{
    auto btype = this->bound_type_;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*2*2);

    int dims = 2;

    for (int idx = 0; idx < dims*fields*N*N; idx++)
    {
        int idx_tmp = idx;
        int f = idx_tmp / (dims * N * N);
        idx_tmp -= f * (dims * N * N);
        int d = idx_tmp / (N * N);
        idx_tmp -= d * (N * N);
        int j = idx_tmp / N;
        idx_tmp -= j * N;
        int i = idx_tmp;

        int iMat, jMat;

        iMat = f*N*N + j*N + i;

        int i_tmp = i;
        for (int offset = -1; offset < 2 && d == 0; offset+=2)
        {
            i = i_tmp + offset;
            if ( (i == -1 || i == N) && btype == ZERO)
                continue;
            else if ( i == -1 )
                i = (N-2);
            else if ( i == N )
                i = 1;
            jMat = f*dims*N*N + d*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,(float) offset));
        }
        i = i_tmp;

        int j_tmp = j;
        for (int offset = -1; offset < 2 && d == 1; offset+=2)
        {
            j = j_tmp + offset;
            if ( (j == -1 || j == N) && btype == ZERO)
                continue;
            else if ( j == -1 )
                j = (N-2);
            else if ( j == N )
                j = 1;
            jMat = f*dims*N*N + d*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,(float) offset));
        }
        j = j_tmp;
    }

    dst = SparseMatrix_(N*N*fields,N*N*2*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::grad(SparseMatrix_ &dst, unsigned int fields)
{
    auto btype = this->bound_type_;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*2*2);

    int dims = 2;

    for (int idx = 0; idx < dims*fields*N*N; idx++)
    {
        int idx_tmp = idx;
        int f = idx_tmp / (dims * N * N);
        idx_tmp -= f * (dims * N * N);
        int d = idx_tmp / (N * N);
        idx_tmp -= d * (N * N);
        int j = idx_tmp / N;
        idx_tmp -= j * N;
        int i = idx_tmp;

        int iMat, jMat;

        iMat = f*dims*N*N + d*N*N + j*N + i;

        int i_tmp = i;
        for (int offset = -1; offset < 2 && d == 0; offset+=2)
        {
            i = i_tmp + offset;
            if ( (i == -1 || i == N) && btype == ZERO)
                continue;
            else if ( i == -1 )
                i = (N-2);
            else if ( i == N )
                i = 1;
            jMat = f*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,(float) offset));
        }
        i = i_tmp;

        int j_tmp = j;
        for (int offset = -1; offset < 2 && d == 1; offset+=2)
        {
            j = j_tmp + offset;
            if ( (j == -1 || j == N) && btype == ZERO)
                continue;
            else if ( j == -1 )
                j = (N-2);
            else if ( j == N )
                j = 1;
            jMat = f*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,(float) offset));
        }
        j = j_tmp;
    }

    dst = SparseMatrix_(N*N*dims*fields,N*N*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}

template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::backstream(Vector_ &dst, const Vector_ &src, const Vector_ &ufield, float dt, FieldType ftype, int fields)
{
    auto btype = this->bound_type_;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    for (int index = 0; index < N*N; index++)
    {
        int j = std::floor(index/N);
        int i = index - N*j;
        Vector_ X(2);
        X(0) = D*(i + .5);
        X(1) = D*(j + .5);

        X = this->sourceTrace(X, ufield, -dt);

        Vector_ interp_indices = (X.array())/D - .5;

        Vector_ interp_quant = calcLinInterp(interp_indices, src, ftype, fields);

        Eigen::VectorXi insert_indices(2);
        insert_indices(0) = i;
        insert_indices(1) = j;

        insertIntoField(insert_indices, interp_quant, dst, ftype, fields);
    }
}

template<int StorageOrder>
JFS_INLINE typename grid2D<StorageOrder>::Vector_ grid2D<StorageOrder>::indexField(Eigen::VectorXi indices, const Vector_ &src, FieldType ftype, int fields)
{
    auto btype = this->bound_type_;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    int i = indices(0);
    int j = indices(1);

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
JFS_INLINE void grid2D<StorageOrder>::insertIntoField(Eigen::VectorXi indices, Vector_ q, Vector_ &dst, FieldType ftype, int fields)
{
    auto btype = this->bound_type_;
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
JFS_INLINE typename grid2D<StorageOrder>::Vector_ grid2D<StorageOrder>::calcLinInterp(Vector_ interp_indices, const Vector_ &src, FieldType ftype, unsigned int fields)
{
    auto btype = this->bound_type_;
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

    Vector_ interp_quant(dims*fields);
    interp_quant.setZero();

    float i0 = interp_indices(0);
    float j0 = interp_indices(1);

    switch (btype)
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

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            int i_tmp = i0_floor + i;
            int j_tmp = j0_floor + j;
            
            float part = std::abs((i_tmp - i0)*(j_tmp - j0));
            i_tmp = (i_tmp == i0_floor) ? i0_ceil : i0_floor;
            j_tmp = (j_tmp == j0_floor) ? j0_ceil : j0_floor;

            if (i_tmp == N || j_tmp == N)
                continue;

            Eigen::VectorXi indices(2);
            indices(0) = i_tmp;
            indices(1) = j_tmp;

            interp_quant = interp_quant.array() + (part * indexField(indices, src, ftype, fields)).array();
        }
    }

    return interp_quant;
}


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::interpolateForce(const std::vector<Force> forces, SparseVector_ &dst)
{
    auto btype = this->bound_type_;
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

