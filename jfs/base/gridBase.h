#ifndef GRIDBASE_H
#define GRIDBASE_H
#include "../jfs_inline.h"

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
class gridBase {
    public:
        gridBase(){}
        
        BOUND_TYPE BOUND;
        
        unsigned int N; // num pixels/voxels per side
        float L; // grid side length
        float D; // pixel/voxel size
        float dt; // time step

        ~gridBase(){}

    protected:        
        
        Eigen::VectorXf X; // position values

        SparseMatrix LAPLACE; // scalar laplace
        SparseMatrix VEC_LAPLACE; // vector laplace
        SparseMatrix DIV; // divergence
        SparseMatrix GRAD; // gradient

        // Linear Interp Stuff
        Eigen::VectorXf ij0; // position values as indices of grid values (not necessarily integers)
        SparseMatrix linInterp; // scalar linear interpolation matrix
        SparseMatrix linInterpVec; // vector linear interpolation matrix

        // set N, L, BOUND, and dt properties
        void initializeGridProperties(unsigned int N, float L, BOUND_TYPE BOUND, float dt);

        // sets X position grid using N, L, D, and dt
        virtual void setXGrid(){}

        // calls initializeGridProperties and setXGrid
        // calculates LAPLACE, VEC_LAPLACE, DIV, and GRAD
        virtual void initializeGrid(unsigned int N, float L, BOUND_TYPE BOUND, float dt){}

        // satisfy boundary conditions for based off BOUND property
        // Inputs:
        //      Eigen::VectorXf &u - velocity field to be updated
        virtual void satisfyBC(Eigen::VectorXf &u){}

        // calculate Laplace operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int dims - dims used to specify scalar vs. vector Laplace
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        virtual void Laplace(SparseMatrix &dst, unsigned int dims, unsigned int fields=1){}

        // calculate Divergence operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. mutliple vector fields concatenated)
        virtual void div(SparseMatrix &dst, unsigned int fields=1){}

        // calculate Gradient operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        virtual void grad(SparseMatrix &dst, unsigned int fields=1){}

        // calculate Interpolation operator from grid values to point
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator (dst*q = value where q is grid quantity interpolated to value)
        //      Eigen::VectorXf &ij0 - grid index values used to interpolate, can be floats
        //      int dims - dimensions of quantity to be interpolated
        //      unsigned int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual void calcLinInterp(SparseMatrix &dst, const Eigen::VectorXf &ij0, int dims, unsigned int fields=1){}
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/base/gridBase.cpp>
#endif

#endif