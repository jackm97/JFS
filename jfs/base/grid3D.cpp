#include <jfs/base/grid3D.h>
#include <iostream>

namespace jfs {

template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::satisfyBC(float* field_data, FieldType ftype, int fields)
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
        dims = 3;
        break;
    }

    int i,j,k;
    if (btype == PERIODIC)
    for (int idx1=0; idx1 < N; idx1++)
    {
        for (int idx2=0; idx2 < N; idx2++)
        {
            for (int f = 0; f < fields; f++)
            {
                for (int d = 0; d < dims; d++)
                {
                    // bottom
                    i = idx1;
                    j = 0;
                    k = idx2;
                    field_data[f*dims*N*N*N + d*N*N*N + N*N*k + N*j + i] = field_data[f*dims*N*N*N + d*N*N*N + N*N*k + N*(N-1) + i];

                    // left
                    i = 0;
                    j = idx1;
                    k = idx2;
                    field_data[f*dims*N*N*N + d*N*N*N + N*N*k + N*j + i] = field_data[f*dims*N*N*N + d*N*N*N + N*N*k + N*j + (N-1)];

                    // front
                    i = idx1;
                    j = idx2;
                    k = 0;
                    field_data[f*dims*N*N*N + d*N*N*N + N*N*k + N*j + i] = field_data[f*dims*N*N*N + d*N*N*N + N*N*(N-1) + N*j + i];
                }
            }
        }
    }

    else if (btype == ZERO)
    for (int idx1=0; idx1 < N; idx1++)
    {
        for (int idx2=0; idx2 < N; idx2++)
        {
            for (int f = 0; f < fields; f++)
            {
                for (int d = 0; d < dims; d++)
                {
                    // bottom
                    i = idx1;
                    j = 0;
                    k = idx2;
                    if (ftype == SCALAR_FIELD || d == 1)
                        field_data[f*dims*N*N*N + d*N*N*N + N*N*k + N*j + i] = 0;
                    
                    // top
                    i = idx1;
                    j = N-1;
                    k = idx2;
                    if (ftype == SCALAR_FIELD || d == 1)
                        field_data[f*dims*N*N*N + d*N*N*N + N*N*k + N*j + i] = 0;

                    // left
                    i = 0;
                    j = idx1;
                    k = idx2;
                    if (ftype == SCALAR_FIELD || d == 0)
                        field_data[f*dims*N*N*N + d*N*N*N + N*N*k + N*j + i] = 0;

                    // right
                    i = N-1;
                    j = idx1;
                    k = idx2;
                    if (ftype == SCALAR_FIELD || d == 0)
                        field_data[f*dims*N*N*N + d*N*N*N + N*N*k + N*j + i] = 0;

                    // front
                    i = idx1;
                    j = idx2;
                    k = 0;
                    if (ftype == SCALAR_FIELD || d == 2)
                        field_data[f*dims*N*N*N + d*N*N*N + N*N*k + N*j + i] = 0;

                    // back
                    i = idx1;
                    j = idx2;
                    k = N-1;
                    if (ftype == SCALAR_FIELD || d == 2)
                        field_data[f*dims*N*N*N + d*N*N*N + N*N*k + N*j + i] = 0;
                }
            }
        }
    }
}


template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::Laplace(SparseMatrix_ &dst, unsigned int dims, unsigned int fields)
{
    auto btype = this->bound_type_;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*5*dims);

    for (int idx = 0; idx < dims*fields*N*N*N; idx++)
    {
        int idx_tmp = idx;
        int f = idx_tmp / (dims * N * N * N);
        idx_tmp -= f * (dims * N * N * N);
        int d = idx_tmp / (N * N * N);
        idx_tmp -= d * (N * N * N);
        int k = idx_tmp / (N * N);
        idx_tmp -= k * N * N;
        int j = idx_tmp / N;
        idx_tmp -= j * N;
        int i = idx_tmp;

        int iMat, jMat;

        iMat = f*dims*N*N*N + d*N*N*N + k*N*N + j*N + i;
        jMat = iMat;
        tripletList.push_back(T(iMat,jMat,-6.f));

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
            jMat = f*dims*N*N*N + d*N*N*N + k*N*N + j*N + i;
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
            jMat = f*dims*N*N*N + d*N*N*N + k*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,1.f));
        }
        j = j_tmp;

        int k_tmp = k;
        for (int offset = -1; offset < 2; offset+=2)
        {
            k = k_tmp + offset;
            if ( (k == -1 || k == N) && btype == ZERO)
                continue;
            else if ( k == -1 )
                k = (N-2);
            else if ( k == N )
                k = 1;
            jMat = f*dims*N*N*N + d*N*N*N + k*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,1.f));
        }
        k = k_tmp;
    }

    dst = SparseMatrix_(N*N*N*dims*fields,N*N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(D*D) * dst;
}


template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::div(SparseMatrix_ &dst, unsigned int fields)
{
    auto btype = this->bound_type_;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*2*2);

    int dims = 3;

    for (int idx = 0; idx < dims*fields*N*N*N; idx++)
    {
        int idx_tmp = idx;
        int f = idx_tmp / (dims * N * N * N);
        idx_tmp -= f * (dims * N * N * N);
        int d = idx_tmp / (N * N * N);
        idx_tmp -= d * (N * N * N);
        int k = idx_tmp / (N * N);
        idx_tmp -= k * N * N;
        int j = idx_tmp / N;
        idx_tmp -= j * N;
        int i = idx_tmp;

        int iMat, jMat;

        iMat = f*N*N*N + k*N*N + j*N + i;

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
            jMat = f*dims*N*N*N + d*N*N*N + k*N*N + j*N + i;
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
            jMat = f*dims*N*N*N + d*N*N*N + k*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,(float) offset));
        }
        j = j_tmp;

        int k_tmp = k;
        for (int offset = -1; offset < 2 && d == 2; offset+=2)
        {
            k = k_tmp + offset;
            if ( (k == -1 || k == N) && btype == ZERO)
                continue;
            else if ( k == -1 )
                k = (N-2);
            else if ( k == N )
                k = 1;
            jMat = f*dims*N*N*N + d*N*N*N + k*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,(float) offset));
        }
        k = k_tmp;
    }

    dst = SparseMatrix_(N*N*N*fields,N*N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}


template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::grad(SparseMatrix_ &dst, unsigned int fields)
{
    auto btype = this->bound_type_;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*2*2);

    int dims = 3;

    for (int idx = 0; idx < dims*fields*N*N*N; idx++)
    {
        int idx_tmp = idx;
        int f = idx_tmp / (dims * N * N * N);
        idx_tmp -= f * (dims * N * N * N);
        int d = idx_tmp / (N * N * N);
        idx_tmp -= d * (N * N * N);
        int k = idx_tmp / (N * N);
        idx_tmp -= k * N * N;
        int j = idx_tmp / N;
        idx_tmp -= j * N;
        int i = idx_tmp;

        int iMat, jMat;

        iMat = f*dims*N*N*N + d*N*N*N + k*N*N + j*N + i;

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
            jMat = f*N*N*N + k*N*N + j*N + i;
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
            jMat = f*N*N*N + k*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,(float) offset));
        }
        j = j_tmp;

        int k_tmp = k;
        for (int offset = -1; offset < 2 && d == 2; offset+=2)
        {
            k = k_tmp + offset;
            if ( (k == -1 || k == N) && btype == ZERO)
                continue;
            else if ( k == -1 )
                k = (N-2);
            else if ( k == N )
                k = 1;
            jMat = f*N*N*N + k*N*N + j*N + i;
            tripletList.push_back(T(iMat,jMat,(float) offset));
        }
        k = k_tmp;
    }

    dst = SparseMatrix_(N*N*N*dims*fields,N*N*N*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;

    if (N == 3)
        std::cout << dst << std::endl;
}

template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::backstream(Vector_ &dst, const Vector_ &src, const Vector_ &ufield, float dt, FieldType ftype, int fields)
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
        dims = 3;
        break;
    }

    for (int index = 0; index < N*N*N; index++)
    {
        int k = std::floor(index/(N*N));
        int j = std::floor((index-N*N*k)/N);
        int i = index - N*N*k - N*j;
        Vector_ X(3);
        X(0) = D*(i + .5);
        X(1) = D*(j + .5);
        X(2) = D*(k + .5);

        using gridBase = gridBase<StorageOrder>;
        X = gridBase::backtrace(X, ufield, -dt);

        Vector_ interp_indices = (X.array())/D - .5;

        Vector_ interp_quant(fields*dims);
        interpGridToPoint(interp_quant.data(), interp_indices.data(), src.data(), ftype, fields);

        Eigen::VectorXi insert_indices(3);
        insert_indices(0) = i;
        insert_indices(1) = j;
        insert_indices(2) = k;

        insertIntoGrid(insert_indices.data(), interp_quant.data(), dst.data(), ftype, fields);
    }
}

template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::
indexGrid(float* dst, int* indices, const float* field_data, FieldType ftype, int fields)
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
        dims = 3;
        break;
    }

    int i = indices[0];
    int j = indices[1];
    int k = indices[2];

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            dst[dims*f + d] = field_data[N*N*N*dims*f + N*N*N*d + N*N*k + N*j + i];
        }
    }
}

template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::
insertIntoGrid(int* indices, float* q, float* field_data, FieldType ftype, int fields)
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
        dims = 3;
        break;
    }

    int i = indices[0];
    int j = indices[1];
    int k = indices[2];

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            field_data[N*N*N*dims*f + N*N*N*d + N*N*k + N*j + i] = q[dims*f + d];
        }
    }
}


template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::
interpGridToPoint(float* dst, float* point, const float* field_data, FieldType ftype, unsigned int fields)
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
        dims = 3;
        break;
    }

    for (int idx = 0; idx < fields*dims; idx++)
        dst[idx] = 0;

    float i0 = point[0];
    float j0 = point[1];
    float k0 = point[2];

    switch (btype)
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
    

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                int i_tmp = i0_floor + i;
                int j_tmp = j0_floor + j;
                int k_tmp = k0_floor + k;
                
                float part = std::abs((i_tmp - i0)*(j_tmp - j0)*(k_tmp - k0));
                i_tmp = (i_tmp == i0_floor) ? i0_ceil : i0_floor;
                j_tmp = (j_tmp == j0_floor) ? j0_ceil : j0_floor;
                k_tmp = (k_tmp == k0_floor) ? k0_ceil : k0_floor;

                if (i_tmp == N || j_tmp == N || k_tmp == N)
                    continue;

                int indices[3];
                indices[0] = i_tmp;
                indices[1] = j_tmp;
                indices[2] = k_tmp;

                float indexed_quant[fields*dims];
                indexGrid(indexed_quant, indices, field_data, ftype, fields);
                for (int idx = 0; idx < fields*dims; idx++)
                    dst[idx] += part*indexed_quant[idx];
            }
        }
    }
}


template<int StorageOrder>
JFS_INLINE void grid3D<StorageOrder>::interpolateForce(const std::vector<Force> forces, SparseVector_ &dst)
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

