#ifndef GRID2D_H
#define GRID2D_H
#include "jfs_inline.h"

#include <jfs/gridBase.h>

namespace jfs {

class grid2D: public gridBase {
    protected:
        grid2D(){}

        void setXGrid();

        void satisfyBC(Eigen::VectorXf &u);

        void Laplace(Eigen::SparseMatrix<float> &dst, unsigned int dims, unsigned int fields=1);

        void div(Eigen::SparseMatrix<float> &dst, unsigned int fields=1);

        void grad(Eigen::SparseMatrix<float> &dst, unsigned int fields=1);

        void calcLinInterp(Eigen::SparseMatrix<float> &dst, const Eigen::VectorXf &ij0, int dims, unsigned int fields=1);

        ~grid2D(){}
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/grid2D.cpp>
#endif

#endif