#ifndef GRIDBASE_H
#define GRIDBASE_H
#include "jfs_inline.h"

namespace jfs {

class gridBase {
    public:
        BOUND_TYPE BOUND;
    protected:        
        unsigned int N; // num pixels/voxels per side
        float L; // grid side length
        float D; // pixel/voxel size
        float dt;
        
        Eigen::VectorXf X;
        Eigen::VectorXf X0;
        Eigen::VectorXf XTemp;

        SparseMatrix LAPLACE;
        SparseMatrix LAPLACEX; // scalar laplace extended for x concatenated fields
        SparseMatrix VEC_LAPLACE;
        SparseMatrix DIV;
        SparseMatrix DIVX; // divergence extended for x concatenated fields
        SparseMatrix GRAD;

        // Linear Interp Stuff
        Eigen::VectorXf ij0;
        SparseMatrix linInterp;
        SparseMatrix linInterpVec;

        gridBase(){}

        virtual void setXGrid(){}

        virtual void satisfyBC(Eigen::VectorXf &u){}

        virtual void Laplace(SparseMatrix &dst, unsigned int dims, unsigned int fields=1){}

        virtual void div(SparseMatrix &dst, unsigned int fields=1){}

        virtual void grad(SparseMatrix &dst, unsigned int fields=1){}

        virtual void calcLinInterp(SparseMatrix &dst, const Eigen::VectorXf &ij0, int dims, unsigned int fields=1){}

        ~gridBase(){}
};
} // namespace jfs

#endif