#ifndef JSSFSOLVERBASE_H
#define JSSFSOLVERBASE_H

#include "jfs_inline.h"

#include <jfs/base/gridBase.h>

#include <Eigen/Eigen>

#include <vector>

namespace jfs {

typedef Eigen::SparseLU< Eigen::SparseMatrix<float> > genSolver; // can solve both boundary conditions
typedef Eigen::SimplicialLDLT< Eigen::SparseMatrix<float> > fastZeroSolver; // solves zero bounds quickly
typedef Eigen::ConjugateGradient< Eigen::SparseMatrix<float>,  Eigen::Lower | Eigen::Upper> iterativeSolver; // iterative solver, great for parallelization

template <class LinearSolver, int StorageOrder>
class JSSFSolverBase : virtual public gridBase{
    public:
        JSSFSolverBase(){};

        virtual void initialize(unsigned int N, float L, BoundType btype, float dt, float visc=0) = 0;

        void resetFluid();

        bool calcNextStep(const std::vector<Force> forces);

        //inline getters
        float* velocityData(){return U0.data();}

        ~JSSFSolverBase(){}
        
    protected:
        typedef typename Eigen::SparseMatrix<float, StorageOrder> SparseMatrix_;
        typedef typename Eigen::Matrix<float, Eigen::Dynamic, 1, StorageOrder> Vector_;

        float visc; // fluid viscosity
        
        SparseMatrix_ ADifU; // ADifU*x = b for diffuseSolveU
        LinearSolver diffuseSolveU;
        LinearSolver projectSolve;
        SparseMatrix_ AProject; // AProject*x = b for projectSolve

        Vector_ b; // b in A*x=b linear equation solve
        Vector_ bVec; // b in A*x=b linear equation solve

        // differential operators
        SparseMatrix_ GRAD;
        SparseMatrix_ DIV;

        Vector_ U; 
        Vector_ U0;
        Vector_ F;

        bool calcNextStep( );

        void addForce(Vector_ &dst, const Vector_ &src, const Vector_ &force, float dt);

        void projection(Vector_ &dst, const Vector_ &src);

        void diffuse(Vector_ &dst, const Vector_ &src, float dt, FieldType ftype);
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/JSSFSolverBase.cpp>
#endif

#endif