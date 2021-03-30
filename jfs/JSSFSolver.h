#ifndef JSSFSOLVER_H
#define JSSFSOLVER_H

#include "jfs_inline.h"

#include <jfs/JSSFSolverBase.h>
#include <jfs/base/grid2D.h>

namespace jfs {

template <class LinearSolver=genSolver, int StorageOrder=Eigen::ColMajor>
class JSSFSolver : virtual public JSSFSolverBase<LinearSolver, StorageOrder>, virtual public grid2D {
    public:
        JSSFSolver(){};

        JSSFSolver(unsigned int N, float L, BoundType btype, float dt, float visc=0);

        void initialize(unsigned int N, float L, BoundType btype, float dt, float visc=0);

        ~JSSFSolver(){}
    protected:

        using SparseMatrix_ = typename JSSFSolverBase<LinearSolver, StorageOrder>::SparseMatrix_;
        using Vector_ = typename JSSFSolverBase<LinearSolver, StorageOrder>::Vector_;
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/JSSFSolver.cpp>
#endif

#endif