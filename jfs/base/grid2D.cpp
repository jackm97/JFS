#include <jfs/base/grid2D.h>

namespace jfs {


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::satisfyBC(float* field_data, FieldType ftype, int fields)
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
                field_data[N*fields*dims*j + fields*dims*i + dims*f + d] = field_data[N*fields*dims*(N-1) + fields*dims*i + dims*f + d];

                // left
                j = idx;
                i = 0;
                field_data[N*fields*dims*j + fields*dims*i + dims*f + d] = field_data[N*fields*dims*j + fields*dims*(N-1) + dims*f + d];
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
                    field_data[N*fields*dims*j + fields*dims*i + dims*f + d] = 0;

                // bottom
                i = idx;
                j = 0;
                if (ftype == SCALAR_FIELD || d == 1)
                    field_data[N*fields*dims*j + fields*dims*i + dims*f + d] = 0;

                // left
                j = idx;
                i = N-1;
                if (ftype == SCALAR_FIELD || d == 0)
                    field_data[N*fields*dims*j + fields*dims*i + dims*f + d] = 0;

                // right
                j = idx;
                i = 0;
                if (ftype == SCALAR_FIELD || d == 0)
                    field_data[N*fields*dims*j + fields*dims*i + dims*f + d] = 0;
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
        int j = idx_tmp / (N*dims*fields);
        idx_tmp -= j * (N*dims*fields);
        int i = idx_tmp / (dims*fields);
        idx_tmp -= i * (dims*fields);
        int f = idx_tmp / dims;
        idx_tmp -= f * dims;
        int d = idx_tmp;

        int iMat, jMat;

        iMat = N*fields*dims*j + fields*dims*i + dims*f + d;
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
            jMat = N*fields*dims*j + fields*dims*i + dims*f + d;
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
            jMat = N*fields*dims*j + fields*dims*i + dims*f + d;
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
        int j = idx_tmp / (N*dims*fields);
        idx_tmp -= j * (N*dims*fields);
        int i = idx_tmp / (dims*fields);
        idx_tmp -= i * (dims*fields);
        int f = idx_tmp / dims;
        idx_tmp -= f * dims;
        int d = idx_tmp;

        int iMat, jMat;

        iMat = N*fields*j + fields*i + f;

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
            jMat = N*fields*dims*j + fields*dims*i + dims*f + d;
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
            jMat = N*fields*dims*j + fields*dims*i + dims*f + d;
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
        int j = idx_tmp / (N*dims*fields);
        idx_tmp -= j * (N*dims*fields);
        int i = idx_tmp / (dims*fields);
        idx_tmp -= i * (dims*fields);
        int f = idx_tmp / dims;
        idx_tmp -= f * dims;
        int d = idx_tmp;

        int iMat, jMat;

        iMat = N*fields*dims*j + fields*dims*i + dims*f + d;

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
            jMat = N*fields*j + fields*i + f;
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
            jMat = N*fields*j + fields*i + f;
            tripletList.push_back(T(iMat,jMat,(float) offset));
        }
        j = j_tmp;
    }

    dst = SparseMatrix_(N*N*dims*fields,N*N*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}

template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::backstream(float* dst_field, const float* src_field, const float* ufield, float dt, FieldType ftype, int fields)
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

    float* interp_quant = new float[fields*dims];

    for (int index = 0; index < N*N; index++)
    {
        int j = std::floor(index/N);
        int i = index - N*j;
        float x[2]{
            D*(i + .5f),
            D*(j + .5f)
        };

        using gridBase = gridBase<StorageOrder>;
        float x_new[2];
        gridBase::backtrace(x_new, x, 2, ufield, -dt);

        float interp_point[2]{
            x_new[0]/D - .5f,
            x_new[1]/D - .5f
        };

        interpGridToPoint(interp_quant, interp_point, src_field, ftype, fields);

        int insert_indices[2]{i, j};

        insertIntoGrid(insert_indices, interp_quant, dst_field, ftype, fields);
    }

    delete [] interp_quant;
}

template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::
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
        dims = 2;
        break;
    }

    int i = indices[0];
    int j = indices[1];
    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            dst[dims*f + d] = field_data[N*fields*dims*j + fields*dims*i + dims*f + d];
        }
    }
}

template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::
insertIntoGrid(int* indices, float* q, float* field_data, FieldType ftype, int fields, InsertType itype)
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

    int i = indices[0];
    int j = indices[1];

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            switch (itype)
            {
            case Replace:
                field_data[N*fields*dims*j + fields*dims*i + dims*f + d] = q[dims*f + d];
                break;
            case Add:
                field_data[N*fields*dims*j + fields*dims*i + dims*f + d] += q[dims*f + d];
                break;
            }
        }
    }
}


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::
interpGridToPoint(float* dst, const float* point, const float* field_data, FieldType ftype, unsigned int fields)
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

    for (int idx = 0; idx < fields*dims; idx++)
        dst[idx] = 0;

    float i0 = point[0];
    float j0 = point[1];

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
    float part;
    

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

            int indices[2];
            indices[0] = i_tmp;
            indices[1] = j_tmp;

            float indexed_quant[fields*dims];
            indexGrid(indexed_quant, indices, field_data, ftype, fields);
            for (int idx = 0; idx < fields*dims; idx++)
                dst[idx] += part*indexed_quant[idx];
        }
    }
}


template<int StorageOrder>
JFS_INLINE void grid2D<StorageOrder>::
interpPointToGrid(const float* q, const float* point, float* field_data, FieldType ftype, unsigned int fields, InsertType itype)
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

    float i0 = point[0];
    float j0 = point[1];

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
    float part;
    float* q_part = new float[dims*fields];
    

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

            int indices[2];
            indices[0] = i_tmp;
            indices[1] = j_tmp;

            for (int idx = 0; idx < fields*dims; idx++)
                q_part[idx] = part*q[idx];

            insertIntoGrid(indices, q_part, field_data, ftype, fields, itype);
        }
    }

    delete [] q_part;
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class grid2D<Eigen::ColMajor>;
template class grid2D<Eigen::RowMajor>;
#endif

} // namespace jfs

