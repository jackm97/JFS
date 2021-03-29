#ifndef GRIDBASE_H
#define GRIDBASE_H
#include "../jfs_inline.h"

#include <jfs/differential_ops/grid_diff2d.h>
#include <jfs/differential_ops/grid_diff3d.h>

namespace jfs {

// Base class for grid based solvers
//
// Grid quantities are represented by 1-D vectors
// 
// For a 2-D NxN grid, the i,j indices represent
// x and y respectively. Indexing scalar quantities
// follows scalar(N*j + i). A vector quantity is indexed
// using vector(N*N*dim + N*j + i) where dim is either 0
// or 1. 
//
// Multiple quantities can be concatenated together.
// For example, if one wants to concatenate three scalar 
// quantities together to represent RGB color values, they
// can specify a scalar array such that scalar(N*N*field + N*j + i)
// where field represents the color from 0-2. Vector quantities
// would be indexed like vector(N*N*2*field + N*N*dim + N*j + i)

enum InsertType{
    Replace,
    Add
};

class gridBase {
    public:
        
        gridBase(){}

        ~gridBase(){}

        template<typename> friend class gridDiff2D;
        template<typename> friend class gridDiff3D;

    protected:
        
        unsigned int N; // num pixels/voxels per side
        float L; // grid side length
        float D; // pixel/voxel size
        float dt; // time step
        
        BoundType bound_type_;

        // calls initializeGridProperties and setXGrid
        // calculates LAPLACE, VEC_LAPLACE, DIV, and GRAD
        void initializeGrid(unsigned int N, float L, BoundType btype, float dt);

        // satisfy boundary conditions for based off BOUND property
        // Inputs:
        //      Vector_ &u - velocity field to be updated
        virtual void satisfyBC(float* field_data, FieldType ftype, int fields=1) = 0;

        // calculate Interpolation operator from grid values to point
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator (dst*q = value where q is grid quantity interpolated to value)
        //      Vector_ &ij0 - grid index values used to interpolate, can be floats
        //      int dims - dimensions of quantity to be interpolated
        //      unsigned int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual void interpGridToPoint(float* q, const float* point, const float* field_data, FieldType ftype, unsigned int fields=1) = 0; 

        // calculate Interpolation operator from grid values to point
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator (dst*q = value where q is grid quantity interpolated to value)
        //      Vector_ &ij0 - grid index values used to interpolate, can be floats
        //      int dims - dimensions of quantity to be interpolated
        //      unsigned int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual void interpPointToGrid(const float* q, const float* point, float* field_data, FieldType ftype, unsigned int fields=1, InsertType itype=Replace) = 0; 

        // backstreams a quantity on the grid
        // Inputs:
        //      Vector_ &dst - destination grid quantity
        //      Vector_ &src - input grid quantity 
        //      Vector_ &u - velocity used to stream quantity
        //      float dt - time step
        //      float dims - dimensions of grid quantity
        virtual void backstream(float* dst, const float* src, const float* ufield, float dt, FieldType ftype, int fields=1) = 0;

        // determines the location of a partice on a grid node at time t+dt
        // Inputs:
        //      Vector_ &X0 - destination at t+dt (X0 is chosen because usually dt is negative)
        //      Vector_ &X - desitination at t 
        //      Vector_ &u - velocity used to stream quantity
        //      float dt - time step
        virtual void backtrace(float* end_point, const float* start_point, int size, const float* ufield, float dt);

        // indexes a scalar or vector field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Vector_ &src - field quantity to index
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        // Returns:
        //      Vector_ q - indexed quantity where q(dims*field + dim) is stored structure
        virtual void indexGrid(float* dst, int* indices, const float* field_data, FieldType ftype, int fields=1) = 0;

        // inserts a scalar or vector into field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Vector_ q - insert quantity where q(dims*field + dim) is stored structure
        //      Vector_ &dst - field quantity inserting into
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual void insertIntoGrid(int* indices, float* q, float* field_data, FieldType ftype, int fields=1, InsertType itype=Replace) = 0;
};

} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/base/gridBase.cpp>
#endif

#endif