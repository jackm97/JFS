#ifndef FLUID2D_H
#define FLUID2D_H
#include "jfs_inline.h"

namespace jfs {

struct Force2D {
    float x=0;
    float y=0;

    float Fx=0;
    float Fy=0;
};

struct Source2D {
    float x=0;
    float y=0;

    ColorRGB color={0,0,0};

    float strength;
};

class fluid2D {
    public:
        void initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dt);

        void resetFluid();
        
        unsigned int N; // num pixels/voxels per side
        float L; // grid side length
        float D; // pixel/voxel size
        BOUND_TYPE BOUND;
        float dt;
        
        Eigen::VectorXf U; 
        Eigen::VectorXf U0;
        Eigen::VectorXf UTemp;
        
        Eigen::VectorXf S;
        Eigen::VectorXf S0;
        Eigen::VectorXf STemp;
        
        Eigen::VectorXf X;
        Eigen::VectorXf X0;
        Eigen::VectorXf XTemp;

        Eigen::SparseVector<float> F;
        Eigen::VectorXf FTemp;
        Eigen::SparseVector<float> SF;
        Eigen::VectorXf SFTemp;

        Eigen::SparseMatrix<float> LAPLACE;
        Eigen::SparseMatrix<float> LAPLACE3;
        Eigen::SparseMatrix<float> VEC_LAPLACE;
        Eigen::SparseMatrix<float> DIV;
        Eigen::SparseMatrix<float> DIV3;
        Eigen::SparseMatrix<float> GRAD;

        // Linear Interp Stuff
        Eigen::VectorXf ij0;
        Eigen::SparseMatrix<float> linInterp;
        Eigen::SparseMatrix<float> linInterpVec;

        void setXGrid();

        void interpolateForce(const std::vector<Force2D> forces);
        
        void interpolateSource( const std::vector<Source2D> sources);

        void satisfyBC(Eigen::VectorXf &u);

        void Laplace(Eigen::SparseMatrix<float> &dst, unsigned int dims, unsigned int fields=1);

        void div(Eigen::SparseMatrix<float> &dst, unsigned int fields=1);

        void grad(Eigen::SparseMatrix<float> &dst, unsigned int fields=1);

        void calcLinInterp(Eigen::SparseMatrix<float> &dst, const Eigen::VectorXf &ij0, int dims, unsigned int fields=1);
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/fluid2D.cpp>
#endif

#endif