#ifndef GRID2D_H
#define GRID2D_H
#include "../jfs_inline.h"

#include <jfs/base/gridBase.h>

namespace jfs {

// gridBase derived class for 2D solvers
class grid2D: public gridBase {
    public:
        grid2D(){dim_type = DIM2;}

        ~grid2D(){}

    protected:

        // sets X position grid using N, L, D, and dt
        virtual void setXGrid();

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

        // calculate Interpolation operator from grid values to point
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator (dst*q = value where q is grid quantity interpolated to value)
        //      Eigen::VectorXf &ij0 - grid index values used to interpolate, can be floats
        //      int dims - dimensions of quantity to be interpolated
        //      unsigned int fields - number of fields of quantity to be interpolated (i.e. scalars concatenated by color channels)
        virtual void calcLinInterp(SparseMatrix &dst, const Eigen::VectorXf &ij0, int dims, unsigned int fields=1);       
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/base/grid2D.cpp>
#endif

#endif