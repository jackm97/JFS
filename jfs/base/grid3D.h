#ifndef GRID3D_H
#define GRID3D_H
#include "../jfs_inline.h"

#include <jfs/base/gridBase.h>

namespace jfs {

class grid3D: public gridBase {

    public:
        grid3D(){dim_type = DIM3;}

        ~grid3D(){}


    protected:

        // calls initializeGridProperties and setXGrid
        // calculates LAPLACE, VEC_LAPLACE, DIV, and GRAD
        virtual void initializeGrid(unsigned int N, float L, BOUND_TYPE BOUND, float dt);

        // satisfy boundary conditions for based off BOUND property
        // Inputs:
        //      Eigen::VectorXf &u - velocity field to be updated
        virtual void satisfyBC(Eigen::VectorXf &u);

        // calculate Laplace operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int dims - dims used to specify scalar vs. vector Laplace
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        virtual void Laplace(SparseMatrix &dst, unsigned int dims, unsigned int fields=1);

        // calculate Divergence operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. mutliple vector fields concatenated)
        virtual void div(SparseMatrix &dst, unsigned int fields=1);

        // calculate Gradient operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        virtual void grad(SparseMatrix &dst, unsigned int fields=1);

        // backstreams a quantity on the grid
        // Inputs:
        //      Eigen::VectorXf &dst - destination grid quantity
        //      Eigen::VectorXf &src - input grid quantity 
        //      Eigen::VectorXf &u - velocity used to stream quantity
        //      float dt - time step
        //      float dims - dimensions of grid quantity
        virtual void backstream(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt, int dims, int fields=1);

        // calculate Interpolation operator from grid values to point
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator (dst*q = value where q is grid quantity interpolated to value)
        //      Eigen::VectorXf &ij0 - grid index values used to interpolate, can be floats
        //      int dims - dimensions of quantity to be interpolated
        //      unsigned int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual Eigen::VectorXf calcLinInterp(Eigen::VectorXf interp_indices, const Eigen::VectorXf &src, int dims, unsigned int fields=1); 

        // indexes a scalar or vector field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Eigen::VectorXf &src - field quantity to index
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        // Returns:
        //      Eigen::VectorXf q - indexed quantity where q(dims*field + dim) is stored structure
        virtual Eigen::VectorXf indexField(Eigen::VectorXi indices, const Eigen::VectorXf &src, int dims, int fields=1);  

        // inserts a scalar or vector into field
        // Inputs:
        //      Eigen::VectorXi indices - indices to index field (i,j,k) (or for 2D only (i,j))
        //      Eigen::VectorXf q - insert quantity where q(dims*field + dim) is stored structure
        //      Eigen::VectorXf &dst - field quantity inserting into
        //      int dims - dimensions of quantity to be interpolated
        //      int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual void insertIntoField(Eigen::VectorXi indices, Eigen::VectorXf q, Eigen::VectorXf &dst, int dims, int fields=1);  
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/grid3D.cpp>
#endif

#endif