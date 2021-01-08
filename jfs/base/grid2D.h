#ifndef GRID2D_H
#define GRID2D_H
#include "../jfs_inline.h"

#include <jfs/base/gridBase.h>

namespace jfs {

// gridBase derived class for 2D solvers
class grid2D: public gridBase {
    protected:
        grid2D(){}

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

        // backstreams a quantity on the grid
        // Inputs:
        //      Eigen::VectorXf &dst - destination grid quantity
        //      Eigen::VectorXf &src - input grid quantity 
        //      Eigen::VectorXf &u - velocity used to stream quantity
        //      float dt - time step
        //      float dims - dimensions of grid quantity
        virtual void backstream(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt, int dims);

        // determines the location of a partice on a grid node at time t+dt
        // Inputs:
        //      Eigen::VectorXf &X0 - destination at t+dt (X0 is chosen because usually dt is negative)
        //      Eigen::VectorXf &X - desitination at t 
        //      Eigen::VectorXf &u - velocity used to stream quantity
        //      float dt - time step
        virtual void sourceTrace(Eigen::VectorXf &X0, const Eigen::VectorXf &X, const Eigen::VectorXf &u, float dt);        

        ~grid2D(){}
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/base/grid2D.cpp>
#endif

#endif