#ifndef JSSFSOLVER_H
#define JSSFSOLVER_H

#include "jfs_inline.h"

#include <jfs/base/fluid2D.h>

#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace jfs {

typedef Eigen::SparseLU< SparseMatrix > genSolver; // can solve both boundary conditions
typedef Eigen::SimplicialLDLT< SparseMatrix > fastZeroSolver; // solves zero bounds quickly
typedef Eigen::ConjugateGradient< SparseMatrix, Eigen::Lower|Eigen::Upper > iterativeSolver; // iterative solver, great for parallelization

template <class LinearSolver=genSolver>
class JSSFSolver : public fluid2D {
    public:
        JSSFSolver(){}

        JSSFSolver(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc=0, float diff=0, float diss=0);

        void initialize(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc=0, float diff=0, float diss=0);

        void calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources);

        ~JSSFSolver(){}

        float visc; // fluid viscosity
        float diff; // particle diffusion
        float diss; // particle dissipation
    private:
        SparseMatrix ADifU; // ADifU*x = b for diffuseSolveU
        LinearSolver diffuseSolveU;
        SparseMatrix ADifS; // ADifS*x = b for diffuseSolveS
        LinearSolver diffuseSolveS;
        LinearSolver projectSolve;

        Eigen::VectorXf b; // b in A*x=b linear equation solve
        Eigen::VectorXf bVec; // b in A*x=b linear equation solve
        Eigen::VectorXf sol; // solution to A*x=b linear equation solve
        Eigen::VectorXf solVec; // solution to A*x=b linear equation solve

        Eigen::VectorXf X0; // position values of particle at X at time t - dt

        void calcNextStep( );

        void addForce(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &force, float dt);

        void transport(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt, int dims);

        void particleTrace(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt);

        void projection(Eigen::VectorXf &dst, const Eigen::VectorXf &src);

        void diffuse(Eigen::VectorXf &dst, const Eigen::VectorXf &src, float dt, int dims);

        void dissipate(Eigen::VectorXf &dst, const Eigen::VectorXf &src, float dt);
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/JSSFSolver.cpp>
#endif

#endif