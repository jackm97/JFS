#include <jfs/base/gridBase.h>
#include <iostream>

namespace jfs
{

template<int StorageOrder>
void gridBase<StorageOrder>::initializeGrid(unsigned int N, float L, BoundType btype, float dt)
{
    this->N = N;
    this->L = L;
    this->D = L/(N-1);
    this->bound_type_ = btype;
    this->dt = dt;
}

template<int StorageOrder>
JFS_INLINE typename gridBase<StorageOrder>::Vector_ gridBase<StorageOrder>::
sourceTrace(Vector_ X, const Vector_ &ufield, int dims, float dt)
{
    Eigen::VectorXi start_indices = ( X.array()/D - .5).template cast<int>();
    Eigen::VectorXf u = indexField(start_indices, ufield, dims);
    Eigen::VectorXf interp_indices = 1/D * (X + u*dt*.5).array() - .5;
    
    u = calcLinInterp(interp_indices, ufield, dims);
    X = X + dt * u;

    return X;
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class gridBase<Eigen::ColMajor>;
template class gridBase<Eigen::RowMajor>;
#endif

} // namespace jfs