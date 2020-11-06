#ifndef GRIDBASE_H
#define GRIDBASE_H
#include "jfs_inline.h"

namespace jfs {

class gridBase {
    protected:        
        unsigned int N; // num pixels/voxels per side
        float L; // grid side length
        float D; // pixel/voxel size
        BOUND_TYPE BOUND;
        float dt;
        
        Eigen::VectorXf X;
        Eigen::VectorXf X0;
        Eigen::VectorXf XTemp;

        Eigen::SparseMatrix<float> LAPLACE;
        Eigen::SparseMatrix<float> LAPLACEX; // scalar laplace extended for x concatenated fields
        Eigen::SparseMatrix<float> VEC_LAPLACE;
        Eigen::SparseMatrix<float> DIV;
        Eigen::SparseMatrix<float> DIVX; // divergence extended for x concatenated fields
        Eigen::SparseMatrix<float> GRAD;

        // Linear Interp Stuff
        Eigen::VectorXf ij0;
        Eigen::SparseMatrix<float> linInterp;
        Eigen::SparseMatrix<float> linInterpVec;

        gridBase(){}

        virtual void setXGrid(){}

        virtual void satisfyBC(Eigen::VectorXf &u){}

        virtual void Laplace(Eigen::SparseMatrix<float> &dst, unsigned int dims, unsigned int fields=1){}

        virtual void div(Eigen::SparseMatrix<float> &dst, unsigned int fields=1){}

        virtual void grad(Eigen::SparseMatrix<float> &dst, unsigned int fields=1){}

        virtual void calcLinInterp(Eigen::SparseMatrix<float> &dst, const Eigen::VectorXf &ij0, int dims, unsigned int fields=1){}

        ~gridBase(){}
};
} // namespace jfs

#endif