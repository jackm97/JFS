#ifndef JSSFSOLVER_H
#define JSSFSOLVER_H

#include "jfs_inline.h"

#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace jfs {

typedef enum {
    ZERO,
    PERIODIC
} BOUND_TYPE;

typedef Eigen::SparseLU< Eigen::SparseMatrix<float> > genSolver; // can solve both boundary conditions
typedef Eigen::SimplicialLDLT< Eigen::SparseMatrix<float> > fastZeroSolver; // solves zero bounds quickly

template <class LinearSolver=genSolver>
class JSSFSolver {
    public:
        Eigen::VectorXf S;
        
        JSSFSolver();

        JSSFSolver(unsigned int N, float L, float D, BOUND_TYPE BOUND, float dt);

        void initialize(unsigned int N, float L, float D, BOUND_TYPE BOUND, float dt);

        void calcNextStep(const Eigen::SparseMatrix<float> &force, const Eigen::SparseMatrix<float> &source, float dt);

        void projection2D(Eigen::VectorXf &dst, const Eigen::VectorXf &src);
    private:
        unsigned int N; // num pixels/voxels per side
        float L; // grid side length
        float D; // pixel/voxel size
        BOUND_TYPE BOUND;
        float dt;
        
        Eigen::VectorXf U; 
        Eigen::VectorXf U0;
        Eigen::VectorXf UTemp;
        Eigen::VectorXf S0;
        Eigen::VectorXf STemp;
        Eigen::VectorXf X;
        Eigen::VectorXf X0;
        Eigen::VectorXf XTemp;

        Eigen::SparseMatrix<float> LAPLACE;
        Eigen::SparseMatrix<float> VEC_LAPLACE;
        Eigen::SparseMatrix<float> DIV;
        Eigen::SparseMatrix<float> GRAD;

        LinearSolver diffuseSolveU;
        LinearSolver diffuseSolveS;
        LinearSolver projectSolve;

        Eigen::VectorXf b; // b in A*x=b linear equation solve
        Eigen::VectorXf bVec; // b in A*x=b linear equation solve
        Eigen::VectorXf sol; // solution to A*x=b linear equation solve
        Eigen::VectorXf solVec; // solution to A*x=b linear equation solve

        // Runge-Kutta Stuff
        Eigen::VectorXf k1;
        Eigen::VectorXf k2;

        // Linear Interp Stuff
        Eigen::VectorXf ij0;
        Eigen::SparseMatrix<float> linInterp;
        Eigen::SparseMatrix<float> linInterpVec;

        void setXGrid2D();

        void addForce(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &force, float dt);

        void transport2D(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt, int dims);

        void particleTrace(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt);

        void diffuse2D(Eigen::VectorXf &dst, const Eigen::VectorXf &src, float dt, int dims);

        void dissipate2D(Eigen::VectorXf &dst, const Eigen::VectorXf &src, float dt);

        void satisfyBC(Eigen::VectorXf &u);

        void Laplace2D(Eigen::SparseMatrix<float> &dst, unsigned int dims);

        void div2D(Eigen::SparseMatrix<float> &dst);

        void grad2D(Eigen::SparseMatrix<float> &dst);

        void calcLinInterp2D(Eigen::SparseMatrix<float> &dst, const Eigen::VectorXf &ij0, int dims);
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/JSSFSolver.cpp>
#endif

#endif