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

template <int StorageOrder>
class gridBase {
    public:
        
        gridBase(){}
        
        BoundType bound_type_;
        
        unsigned int N; // num pixels/voxels per side
        float L; // grid side length
        float D; // pixel/voxel size
        float dt; // time step

        ~gridBase(){}

    protected:
        typedef Eigen::SparseMatrix<float, StorageOrder> SparseMatrix_;
        typedef Eigen::SparseVector<float, StorageOrder> SparseVector_;
        typedef Eigen::VectorXf Vector_;

        // calls initializeGridProperties and setXGrid
        // calculates LAPLACE, VEC_LAPLACE, DIV, and GRAD
        void initializeGrid(unsigned int N, float L, BoundType btype, float dt);

        // satisfy boundary conditions for based off BOUND property
        // Inputs:
        //      Vector_ &u - velocity field to be updated
        virtual void satisfyBC(Vector_ &dst, FieldType ftype, int fields=1) = 0;

        // calculate Laplace operator
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator
        //      unsigned int dims - dims used to specify scalar vs. vector Laplace
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        virtual void Laplace(SparseMatrix_ &dst, unsigned int dims, unsigned int fields=1) = 0;

        // calculate Divergence operator
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. mutliple vector fields concatenated)
        virtual void div(SparseMatrix_ &dst, unsigned int fields=1) = 0;

        // calculate Gradient operator
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        virtual void grad(SparseMatrix_ &dst, unsigned int fields=1) = 0;

        // calculate Interpolation operator from grid values to point
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator (dst*q = value where q is grid quantity interpolated to value)
        //      Vector_ &ij0 - grid index values used to interpolate, can be floats
        //      int dims - dimensions of quantity to be interpolated
        //      unsigned int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual Vector_ calcLinInterp(Vector_ interp_indices, const Vector_ &src, FieldType ftype, unsigned int fields=1) = 0;

        // backstreams a quantity on the grid
        // Inputs:
        //      Vector_ &dst - destination grid quantity
        //      Vector_ &src - input grid quantity 
        //      Vector_ &u - velocity used to stream quantity
        //      float dt - time step
        //      float dims - dimensions of grid quantity
        virtual void backstream(Vector_ &dst, const Vector_ &src, const Vector_ &u, float dt, FieldType ftype, int fields=1) = 0;

        // determines the location of a partice on a grid node at time t+dt
        // Inputs:
        //      Vector_ &X0 - destination at t+dt (X0 is chosen because usually dt is negative)
        //      Vector_ &X - desitination at t 
        //      Vector_ &u - velocity used to stream quantity
        //      float dt - time step
        virtual Vector_ sourceTrace(Vector_ X, const Vector_ &ufield, float dt);

        // indexes a scalar or vector field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Vector_ &src - field quantity to index
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        // Returns:
        //      Vector_ q - indexed quantity where q(dims*field + dim) is stored structure
        virtual Vector_ indexField(Eigen::VectorXi indices, const Vector_ &src, FieldType ftype, int fields=1) = 0;

        // inserts a scalar or vector into field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Vector_ q - insert quantity where q(dims*field + dim) is stored structure
        //      Vector_ &dst - field quantity inserting into
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual void insertIntoField(Eigen::VectorXi indices, Vector_ q, Vector_ &dst, FieldType ftype, int fields=1) = 0;

        virtual void interpolateForce(const std::vector<Force> forces, SparseVector_ &dst) = 0;
        
        virtual void interpolateSource(const std::vector<Source> sources, SparseVector_ &dst) = 0;
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/base/gridBase.cpp>
#endif

#endif