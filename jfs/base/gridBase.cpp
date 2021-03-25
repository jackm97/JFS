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
JFS_INLINE void gridBase<StorageOrder>::
backtrace(float* end_point, const float* start_point, int size, const float* ufield, float dt)
{
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
}

// explicit instantiation of templates
#ifdef JFS_STATIC
template class gridBase<Eigen::ColMajor>;
template class gridBase<Eigen::RowMajor>;
#endif

} // namespace jfs