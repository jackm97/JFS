#ifndef GRID_DIFF3D_H
#define GRID_DIFF3D_H
#include "../jfs_inline.h"

#include <jfs/grid/grid3D.h>

namespace jfs {

// forward declarations
class gridBase;

template <class SparseMatrix>
class gridDiff3D {

public:

        // calculate Laplace operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int dims - dims used to specify scalar vs. vector Laplace
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        static void Laplace(const grid3D* grid, SparseMatrix &dst, unsigned int dims, unsigned int fields=1);

        // calculate Divergence operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. mutliple vector fields concatenated)
        static void div(const grid3D* grid, SparseMatrix &dst, unsigned int fields=1);

        // calculate Gradient operator
        // Inputs:
        //      SparseMatrix &dst - destination sparse matrix for operator
        //      unsigned int fields - if there are multiple fields to concatenate (i.e. scalars concatenated by color channels)
        static void grad(const grid3D* grid, SparseMatrix &dst, unsigned int fields=1);
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/base/grid_dff3d.cpp>
#endif

#endif