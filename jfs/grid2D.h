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

        void Laplace(SparseMatrix &dst, unsigned int dims, unsigned int fields=1);

        void div(SparseMatrix &dst, unsigned int fields=1);

        void grad(SparseMatrix &dst, unsigned int fields=1);

        void calcLinInterp(SparseMatrix &dst, const Eigen::VectorXf &ij0, int dims, unsigned int fields=1);

        ~grid2D(){}
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/grid2D.cpp>
#endif

#endif