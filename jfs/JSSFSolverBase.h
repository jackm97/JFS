#ifndef JSSFSOLVERBASE_H
#define JSSFSOLVERBASE_H

#include "jfs_inline.h"

#include <jfs/base/fluidBase.h>
#include <jfs/base/gridBase.h>

namespace jfs {

typedef Eigen::SparseLU< Eigen::SparseMatrix<float> > genSolver; // can solve both boundary conditions
typedef Eigen::SimplicialLDLT< Eigen::SparseMatrix<float> > fastZeroSolver; // solves zero bounds quickly
typedef Eigen::ConjugateGradient< Eigen::SparseMatrix<float>,  Eigen::Lower | Eigen::Upper> iterativeSolver; // iterative solver, great for parallelization

template <class LinearSolver, int StorageOrder>
class JSSFSolverBase : virtual public fluidBase, virtual public gridBase<StorageOrder>{
    public:
        JSSFSolverBase(){};

        virtual void initialize(unsigned int N, float L, BOUND_TYPE BOUND, float dt, float visc=0, float diff=0, float diss=0) = 0;

        void resetFluid();

        bool calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources);

        ~JSSFSolverBase(){}

        float visc; // fluid viscosity
        float diff; // particle diffusion
        float diss; // particle dissipation
    protected:
        using SparseMatrix_ = typename gridBase<StorageOrder>::SparseMatrix_;
        using SparseVector_ = typename gridBase<StorageOrder>::SparseVector_;
        using Vector_ = typename gridBase<StorageOrder>::Vector_;
        
        SparseMatrix_ ADifU; // ADifU*x = b for diffuseSolveU
        LinearSolver diffuseSolveU;
        SparseMatrix_ ADifS; // ADifS*x = b for diffuseSolveS
        LinearSolver diffuseSolveS;
        LinearSolver projectSolve;
        SparseMatrix_ AProject; // AProject*x = b for projectSolve

        Vector_ b; // b in A*x=b linear equation solve
        Vector_ bVec; // b in A*x=b linear equation solve

        // differential operators
        SparseMatrix_ GRAD;
        SparseMatrix_ DIV;

        Vector_ U; 
        Vector_ U0;
        
        Vector_ S;
        Vector_ S0;

        SparseVector_ F;
        SparseVector_ SF;

        bool calcNextStep( );

        void addForce(Vector_ &dst, const Vector_ &src, const Vector_ &force, float dt);

        void projection(Vector_ &dst, const Vector_ &src);

        void diffuse(Vector_ &dst, const Vector_ &src, float dt, FIELD_TYPE ftype);

        void dissipate(Vector_ &dst, const Vector_ &src, float dt);
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/JSSFSolverBase.cpp>
#endif

#endif