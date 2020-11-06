#ifndef JSSFSOLVER_H
#define JSSFSOLVER_H

#include "jfs_inline.h"

#include <jfs/fluid2D.h>

#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace jfs {

typedef Eigen::SparseLU< Eigen::SparseMatrix<float> > genSolver; // can solve both boundary conditions
typedef Eigen::SimplicialLDLT< Eigen::SparseMatrix<float> > fastZeroSolver; // solves zero bounds quickly

template <class LinearSolver=genSolver>
class JSSFSolver : protected fluid2D {
    public:
        JSSFSolver();

        JSSFSolver(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc=0, float diff=0, float diss=0);

        void initialize(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc=0, float diff=0, float diss=0);

        void calcNextStep();

        void calcNextStep(const std::vector<Force2D> forces, const std::vector<Source2D> sources);

        void getImage(Eigen::VectorXf &image);

        float visc; // fluid viscosity
        float diff; // particle diffusion
        float diss; // particle dissipation
    private:
        LinearSolver diffuseSolveU;
        LinearSolver diffuseSolveS;
        LinearSolver projectSolve;

        Eigen::VectorXf b; // b in A*x=b linear equation solve
        Eigen::VectorXf bVec; // b in A*x=b linear equation solve
        Eigen::VectorXf sol; // solution to A*x=b linear equation solve
        Eigen::VectorXf solVec; // solution to A*x=b linear equation solve

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