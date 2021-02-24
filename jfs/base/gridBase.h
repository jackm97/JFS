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

typedef enum {
    BASE = -1,
    DIM2 = 2,
    DIM3 = 3
} DIMENSION_TYPE;

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

        // make sure to change in child class
        DIMENSION_TYPE dim_type = BASE;        
        
        // Eigen::VectorXf X; // position values
        // Eigen::VectorXf X0; // position values of particle at X at time t - dt (used for advecting quantities)

        SparseMatrix LAPLACE; // scalar laplace
        SparseMatrix VEC_LAPLACE; // vector laplace
        SparseMatrix DIV; // divergence
        SparseMatrix GRAD; // gradient

        // Linear Interp Stuff
        // Eigen::VectorXf ij0; // position values as indices of grid values (not necessarily integers)
        // SparseMatrix linInterp; // scalar linear interpolation matrix
        // SparseMatrix linInterpVec; // vector linear interpolation matrix

        // set N, L, BOUND, and dt properties
        virtual void initializeGridProperties(unsigned int N, float L, BOUND_TYPE BOUND, float dt);

        // calls initializeGridProperties and setXGrid
        // calculates LAPLACE, VEC_LAPLACE, DIV, and GRAD
        virtual void initializeGrid(unsigned int N, float L, BOUND_TYPE BOUND, float dt) = 0;

        // satisfy boundary conditions for based off BOUND property
        // Inputs:
        //      Eigen::VectorXf &u - velocity field to be updated
        virtual void satisfyBC(Eigen::VectorXf &u) = 0;

        // calculate Laplace operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int dims - dims used to specify scalar vs. vector Laplace
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        virtual void Laplace(SparseMatrix &dst, unsigned int dims, unsigned int fields=1) = 0;

        // calculate Divergence operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. mutliple vector fields concatenated)
        virtual void div(SparseMatrix &dst, unsigned int fields=1) = 0;

        // calculate Gradient operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        virtual void grad(SparseMatrix &dst, unsigned int fields=1) = 0;

        // calculate Interpolation operator from grid values to point
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator (dst*q = value where q is grid quantity interpolated to value)
        //      Eigen::VectorXf &ij0 - grid index values used to interpolate, can be floats
        //      int dims - dimensions of quantity to be interpolated
        //      unsigned int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual Eigen::VectorXf calcLinInterp(Eigen::VectorXf interp_indices, const Eigen::VectorXf &src, int dims, unsigned int fields=1) = 0;

        // backstreams a quantity on the grid
        // Inputs:
        //      Eigen::VectorXf &dst - destination grid quantity
        //      Eigen::VectorXf &src - input grid quantity 
        //      Eigen::VectorXf &u - velocity used to stream quantity
        //      float dt - time step
        //      float dims - dimensions of grid quantity
        virtual void backstream(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt, int dims, int fields=1) = 0;

        // determines the location of a partice on a grid node at time t+dt
        // Inputs:
        //      Eigen::VectorXf &X0 - destination at t+dt (X0 is chosen because usually dt is negative)
        //      Eigen::VectorXf &X - desitination at t 
        //      Eigen::VectorXf &u - velocity used to stream quantity
        //      float dt - time step
        virtual Eigen::VectorXf sourceTrace(Eigen::VectorXf X, const Eigen::VectorXf &ufield, int dims, float dt);

        // indexes a scalar or vector field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Eigen::VectorXf &src - field quantity to index
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        // Returns:
        //      Eigen::VectorXf q - indexed quantity where q(dims*field + dim) is stored structure
        virtual Eigen::VectorXf indexField(Eigen::VectorXi indices, const Eigen::VectorXf &src, int dims, int fields=1) = 0;

        // inserts a scalar or vector into field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Eigen::VectorXf q - insert quantity where q(dims*field + dim) is stored structure
        //      Eigen::VectorXf &dst - field quantity inserting into
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual void insertIntoField(Eigen::VectorXi indices, Eigen::VectorXf q, Eigen::VectorXf &dst, int dims, int fields=1) = 0;
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/base/gridBase.cpp>
#endif

#endif