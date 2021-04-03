#ifndef CUDA_GRID2D_H
#define CUDA_GRID2D_H
#include "../../jfs_inline.h"

#ifndef __CUDA_ARCH__
    #define __HOST__DEVICE__
#else
    #define __HOST__DEVICE__ __host__ __device__
#endif

namespace jfs {

class cudaGrid2D {
    public:
        __HOST__DEVICE__
        cudaGrid2D(){}

        __HOST__DEVICE__
        ~cudaGrid2D(){}

    #ifndef __CUDA_ARCH__
    protected:
    #else
    public:
    #endif
        
        unsigned int N; // num pixels/voxels per side
        float L; // grid side length
        float D; // pixel/voxel size
        float dt; // time step
        
        BoundType bound_type_;

        // calls initializeGridProperties and setXGrid
        // calculates LAPLACE, VEC_LAPLACE, DIV, and GRAD
        __HOST__DEVICE__
        void initializeGrid(unsigned int N, float L, BoundType btype, float dt);

        // satisfy boundary conditions for based off BOUND property
        // Inputs:
        //      Vector_ &u - velocity field to be updated
        __HOST__DEVICE__
        void satisfyBC(float* field_data, FieldType ftype, int fields=1);

        // calculate Interpolation operator from grid values to point
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator (dst*q = value where q is grid quantity interpolated to value)
        //      Vector_ &ij0 - grid index values used to interpolate, can be floats
        //      int dims - dimensions of quantity to be interpolated
        //      unsigned int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels
        __HOST__DEVICE__
        void interpGridToPoint(float* dst, const float* point, const float* field_data, FieldType ftype, unsigned int fields=1);

        // calculate Interpolation operator from grid values to point
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator (dst*q = value where q is grid quantity interpolated to value)
        //      Vector_ &ij0 - grid index values used to interpolate, can be floats
        //      int dims - dimensions of quantity to be interpolated
        //      unsigned int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        __HOST__DEVICE__
        void interpPointToGrid(const float* q, const float* point, float* field_data, FieldType ftype, unsigned int fields=1, InsertType itype=Replace); 

        // indexes a scalar or vector field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Vector_ &src - field quantity to index
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        // Returns:
        //      Vector_ q - indexed quantity where q(dims*field + dim) is stored structure
        __HOST__DEVICE__
        void indexGrid(float* dst, int* indices, const float* field_data, FieldType ftype, int fields=1); 

        // inserts a scalar or vector into field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Vector_ q - insert quantity where q(dims*field + dim) is stored structure
        //      Vector_ &dst - field quantity inserting into
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        __HOST__DEVICE__
        void insertIntoGrid(int* indices, float* q, float* field_data, FieldType ftype, int fields=1, InsertType itype=Replace);

        // backstreams a quantity on the grid
        // Inputs:
        //      Vector_ &dst - destination grid quantity
        //      Vector_ &src - input grid quantity 
        //      Vector_ &u - velocity used to stream quantity
        //      float dt - time step
        //      float dims - dimensions of grid quantity
        __HOST__DEVICE__
        void backstream(float* dst, const float* src, const float* ufield, float dt, FieldType ftype, int fields=1);

        // determines the location of a partice on a grid node at time t+dt
        // Inputs:
        //      Vector_ &X0 - destination at t+dt (X0 is chosen because usually dt is negative)
        //      Vector_ &X - desitination at t 
        //      Vector_ &u - velocity used to stream quantity
        //      float dt - time step
        __HOST__DEVICE__
        void backtrace(float* end_point, const float* start_point, const float* ufield, float dt);
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/cuda/grid/cuda_grid2d.cu>
#endif

#endif