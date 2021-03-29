#include <jfs/differential_ops/grid_diff3d.h>

#include <Eigen/Eigen>

namespace jfs {

template <class SparseMatrix>
JFS_INLINE void gridDiff3D<SparseMatrix>::Laplace(const gridBase* grid, SparseMatrix &dst, unsigned int dims, unsigned int fields)
{
    auto btype = grid->bound_type_;
    auto L = grid->L;
    auto N = grid->N;
    auto D = grid->D;

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*5*dims);

    for (int idx = 0; idx < dims*fields*N*N*N; idx++)
    {
        int idx_tmp = idx;
        int k = idx_tmp / (N*N*dims*fields);
        idx_tmp -= k * (N*N*dims*fields);
        int j = idx_tmp / (N*dims*fields);
        idx_tmp -= j * (N*dims*fields);
        int i = idx_tmp / (dims*fields);
        idx_tmp -= i * (dims*fields);
        int f = idx_tmp / dims;
        idx_tmp -= f * dims;
        int d = idx_tmp;

        int iMat, jMat;

        iMat = N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d;
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
            jMat = N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d;
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
            jMat = N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d;
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
            jMat = N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d;
            tripletList.push_back(T(iMat,jMat,1.f));
        }
        k = k_tmp;
    }

    dst = SparseMatrix(N*N*N*dims*fields,N*N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(D*D) * dst;
}


template <class SparseMatrix>
JFS_INLINE void gridDiff3D<SparseMatrix>::div(const gridBase* grid, SparseMatrix &dst, unsigned int fields)
{
    auto btype = grid->bound_type_;
    auto L = grid->L;
    auto N = grid->N;
    auto D = grid->D;

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

        iMat = N*N*fields*k + N*fields*j + fields*i + f;

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
            jMat = N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d;
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
            jMat = N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d;
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
            jMat = N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d;
            tripletList.push_back(T(iMat,jMat,(float) offset));
        }
        k = k_tmp;
    }

    dst = SparseMatrix(N*N*N*fields,N*N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}


template <class SparseMatrix>
JFS_INLINE void gridDiff3D<SparseMatrix>::grad(const gridBase* grid, SparseMatrix &dst, unsigned int fields)
{
    auto btype = grid->bound_type_;
    auto L = grid->L;
    auto N = grid->N;
    auto D = grid->D;

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

        iMat = N*N*fields*dims*k + N*fields*dims*j + fields*dims*i + dims*f + d;

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
            jMat = N*N*fields*k + N*fields*j + fields*i + f;
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
            jMat = N*N*fields*k + N*fields*j + fields*i + f;
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
            jMat = N*N*fields*k + N*fields*j + fields*i + f;
            tripletList.push_back(T(iMat,jMat,(float) offset));
        }
        k = k_tmp;
    }

    dst = SparseMatrix(N*N*N*dims*fields,N*N*N*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}


// explicit instantiation of templates
#ifdef JFS_STATIC
template class gridDiff3D<Eigen::SparseMatrix<float, Eigen::ColMajor>>;
template class gridDiff3D<Eigen::SparseMatrix<float, Eigen::RowMajor>>;
#endif

} // namespace jfs