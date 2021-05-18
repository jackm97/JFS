#include <jfs/grid/grid3D.h>

#include <cmath>

namespace jfs {

JFS_INLINE void grid3D::initializeGrid(unsigned int N, float L, BoundType btype, float dt)
{
    this->N = N;
    this->L = L;
    this->D = L/(N-1);
    this->bound_type_ = btype;
    this->dt = dt;
}

JFS_INLINE void grid3D::satisfyBC(float* field_data, FieldType ftype, int fields)
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
                    field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d] = field_data[N*N*fields*dims*k + N*fields*dims*(N-1) + fields*dims*i + dims*f + d];

                    // left
                    i = 0;
                    j = idx1;
                    k = idx2;
                    field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d] = field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*(N-1) + dims*f + d];

                    // front
                    i = idx1;
                    j = idx2;
                    k = 0;
                    field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d] = field_data[N*N*fields*dims*(N-1) + N*fields*dims*j + fields*dims*i + dims*f + d];
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
                        field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d] = 0;
                    
                    // top
                    i = idx1;
                    j = N-1;
                    k = idx2;
                    if (ftype == SCALAR_FIELD || d == 1)
                        field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d] = 0;

                    // left
                    i = 0;
                    j = idx1;
                    k = idx2;
                    if (ftype == SCALAR_FIELD || d == 0)
                        field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d] = 0;

                    // right
                    i = N-1;
                    j = idx1;
                    k = idx2;
                    if (ftype == SCALAR_FIELD || d == 0)
                        field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d] = 0;

                    // front
                    i = idx1;
                    j = idx2;
                    k = 0;
                    if (ftype == SCALAR_FIELD || d == 2)
                        field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d] = 0;

                    // back
                    i = idx1;
                    j = idx2;
                    k = N-1;
                    if (ftype == SCALAR_FIELD || d == 2)
                        field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d] = 0;
                }
            }
        }
    }
}

JFS_INLINE void grid3D::backstream(float* dst_field, const float* src_field, const float* ufield, float dt, FieldType ftype, int fields)
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

    float* interp_quant = new float[fields*dims];

    for (int index = 0; index < N*N; index++)
    {
        int k = index/(N*N);
        int j = (index-N*N*k)/N;
        int i = index - N*N*k - N*j;
        float x[3]{
            D*(i + .5f),
            D*(j + .5f),
            D*(k + .5f)
        };

        float x_new[3];
        backtrace(x_new, x, ufield, -dt);

        float interp_point[3]{
            x_new[0]/D - .5f,
            x_new[1]/D - .5f,
            x_new[2]/D - .5f
        };

        interpGridToPoint(interp_quant, interp_point, src_field, ftype, fields);

        int insert_indices[3]{i, j, k};

        insertIntoGrid(insert_indices, interp_quant, dst_field, ftype, fields);
    }

    delete [] interp_quant;
}

JFS_INLINE void grid3D::
backtrace(float* end_point, const float* start_point, const float* ufield, float dt)
{
    int size = 3;

    int* start_indices = new int[size];
    for (int i = 0; i < size; i++)
        start_indices[i] = (int) ( start_point[i]/D - .5 );
    
    float* u = new float[size];
    indexGrid(u, start_indices, ufield, VECTOR_FIELD);
    
    float* interp_indices = new float[size];
    for (int i = 0; i < size; i++)
        interp_indices[i] = 1/D * ( (float)start_point[i] + u[i] * dt/2) - .5;

    
    interpGridToPoint(u, interp_indices, ufield, VECTOR_FIELD);
    for (int i = 0; i < size; i++)
        end_point[i] = start_point[i] + dt * u[i];

    delete [] start_indices;
    delete [] u;
    delete [] interp_indices;
}

JFS_INLINE void grid3D::
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
            dst[dims*f + d] = field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + f];
        }
    }
}

JFS_INLINE void grid3D::
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
            switch (itype)
            {
            case Replace:
                field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + f] = q[dims*f + d];
                break;
            
            case Add:
                field_data[N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + f] += q[dims*f + d];
                break;
            }
        }
    }
}


JFS_INLINE void grid3D::
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


JFS_INLINE void grid3D::
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
        dims = 3;
        break;
    }

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
    float* q_part = new float[dims*fields];
    

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

                for (int idx = 0; idx < fields*dims; idx++)
                    q_part[idx] = part*q[idx];

                insertIntoGrid(indices, q_part, field_data, ftype, fields, itype);
            }
        }
    }

    delete [] q_part;
}

} // namespace jfs

