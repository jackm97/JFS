#ifndef JSSFSOLVER_H
#define JSSFSOLVER_H

#include "jfs_inline.h"

#include <jfs/JSSFSolverBase.h>
#include <jfs/base/grid2D.h>

namespace jfs {

template <class LinearSolver=genSolver, int StorageOrder=Eigen::ColMajor>
class JSSFSolver : virtual public JSSFSolverBase<LinearSolver, StorageOrder>, virtual public grid2D<StorageOrder> {
    public:
        JSSFSolver(){};

        JSSFSolver(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc=0, float diff=0, float diss=0);

        void initialize(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc=0, float diff=0, float diss=0);

        void getImage(Eigen::VectorXf &img);

        ~JSSFSolver(){}
    protected:

        using SparseMatrix_ = typename JSSFSolverBase<LinearSolver, StorageOrder>::SparseMatrix_;
        using SparseVector_ = typename JSSFSolverBase<LinearSolver, StorageOrder>::SparseVector_;
        using Vector_ = typename JSSFSolverBase<LinearSolver, StorageOrder>::Vector_;
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/JSSFSolver.cpp>
#endif

#endif