#ifndef JSSFSOLVER3D_H
#define JSSFSOLVER3D_H

#include "jfs_inline.h"

#include <jfs/JSSFSolverBase.h>
#include <jfs/base/grid3D.h>

namespace jfs {

template <class LinearSolver=genSolver, int StorageOrder=Eigen::ColMajor>
class JSSFSolver3D : virtual public JSSFSolverBase<LinearSolver, StorageOrder>, virtual public grid3D<StorageOrder> {
    public:
        JSSFSolver3D(){};

        JSSFSolver3D(unsigned int N, float L, BoundType btype, float dt, float visc=0, float diff=0, float diss=0);

        void initialize(unsigned int N, float L, BoundType btype, float dt, float visc=0, float diff=0, float diss=0);

        void getImage(Eigen::VectorXf &img);

        ~JSSFSolver3D(){}
    protected:

        using SparseMatrix_ = typename JSSFSolverBase<LinearSolver, StorageOrder>::SparseMatrix_;
        using SparseVector_ = typename JSSFSolverBase<LinearSolver, StorageOrder>::SparseVector_;
        using Vector_ = typename JSSFSolverBase<LinearSolver, StorageOrder>::Vector_;
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/JSSFSolver3D.cpp>
#endif

#endif