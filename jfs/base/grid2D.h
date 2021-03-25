#ifndef GRID2D_H
#define GRID2D_H
#include "../jfs_inline.h"

#include <jfs/base/gridBase.h>

namespace jfs {

// gridBase derived class for 2D solvers
template <int StorageOrder>
class grid2D: virtual public gridBase<StorageOrder> {
    public:
        grid2D(){}

        ~grid2D(){}

    protected:

        using SparseMatrix_ = typename gridBase<StorageOrder>::SparseMatrix_;
        using SparseVector_ = typename gridBase<StorageOrder>::SparseVector_;
        using Vector_ = typename gridBase<StorageOrder>::Vector_;

        // satisfy boundary conditions for based off BOUND property
        // Inputs:
        //      Vector_ &u - velocity field to be updated
        virtual void satisfyBC(float* field_data, FieldType ftype, int fields=1);

        // calculate Laplace operator
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator
        //      unsigned int dims - dims used to specify scalar vs. vector Laplace
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        virtual void Laplace(SparseMatrix_ &dst, unsigned int dims, unsigned int fields=1);

        // calculate Divergence operator
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. mutliple vector fields concatenated)
        virtual void div(SparseMatrix_ &dst, unsigned int fields=1);

        // calculate Gradient operator
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        virtual void grad(SparseMatrix_ &dst, unsigned int fields=1);

        // backstreams a quantity on the grid
        // Inputs:
        //      Vector_ &dst - destination grid quantity
        //      Vector_ &src - input grid quantity 
        //      Vector_ &u - velocity used to stream quantity
        //      float dt - time step
        //      float dims - dimensions of grid quantity
        virtual void backstream(float* dst, const float* src, const float* ufield, float dt, FieldType ftype, int fields=1);

        // calculate Interpolation operator from grid values to point
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator (dst*q = value where q is grid quantity interpolated to value)
        //      Vector_ &ij0 - grid index values used to interpolate, can be floats
        //      int dims - dimensions of quantity to be interpolated
        //      unsigned int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual void interpGridToPoint(float* dst, const float* point, const float* field_data, FieldType ftype, unsigned int fields=1);

        // calculate Interpolation operator from grid values to point
        // Inputs:
        //      SparseMatrix_ &dst - destination sparse matrix for operator (dst*q = value where q is grid quantity interpolated to value)
        //      Vector_ &ij0 - grid index values used to interpolate, can be floats
        //      int dims - dimensions of quantity to be interpolated
        //      unsigned int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual void interpPointToGrid(const float* q, const float* point, float* field_data, FieldType ftype, unsigned int fields=1, InsertType itype=Replace); 

        // indexes a scalar or vector field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Vector_ &src - field quantity to index
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        // Returns:
        //      Vector_ q - indexed quantity where q(dims*field + dim) is stored structure
        virtual void indexGrid(float* dst, int* indices, const float* field_data, FieldType ftype, int fields=1); 

        // inserts a scalar or vector into field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Vector_ q - insert quantity where q(dims*field + dim) is stored structure
        //      Vector_ &dst - field quantity inserting into
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual void insertIntoGrid(int* indices, float* q, float* field_data, FieldType ftype, int fields=1, InsertType itype=Replace); 
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/base/grid2D.cpp>
#endif

#endif