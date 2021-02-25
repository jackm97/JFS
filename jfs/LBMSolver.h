#ifndef LBMSOLVER_H
#define LBMSOLVER_H

#include "jfs_inline.h"

#include <jfs/base/fluidBase.h>
#include <jfs/base/grid2D.h>

#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace jfs {

class LBMSolver : public fluidBase, public grid2D<Eigen::ColMajor> {

    public:
        // typical density of fluid
        float rho0;
        // fluid viscosity
        float visc; 
        // speed of sound of fluid
        float us;  
        float dt; // physical time step

        LBMSolver(){};
        
        LBMSolver(unsigned int N, float L, float fps, float rho0=1.3, float visc = 1e-4, float uref = 1);

        void initialize(unsigned int N, float L, float fps, float rho0=1.3, float visc = 1e-4, float uref = 1);

        void resetFluid();

        void getImage(Eigen::VectorXf &img);

        bool calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources);

        ~LBMSolver(){};

    private:

        using SparseMatrix_ = typename grid2D<Eigen::ColMajor>::SparseMatrix_;
        using SparseVector_ = typename grid2D<Eigen::ColMajor>::SparseVector_;
        using Vector_ = typename grid2D<Eigen::ColMajor>::Vector_;

        // DEV NOTES:
        // The paper uses fi to represent the discretized distribution function
        // Because of this, j,k,l are now used to index x,y,z positions on the grid

        Vector_ f; // the distribution function
        Vector_ rho; // calculated rho from distribution function

        Vector_ U; 
        
        Vector_ S;
        Vector_ S0;

        SparseVector_ F;
        SparseVector_ SF;

        Eigen::Vector2f c[9] = { // D2Q9 velocity dicretization
            {0,0},                                // i = 0
            {1,0}, {-1,0}, {0,1}, {0,-1},   // i = 1, 2, 3, 4
            {1,1}, {-1,1}, {1,-1}, {-1,-1}  // i = 5, 6, 7, 8
        };

        float w[9] = { // lattice weights
            4./9.,                          // i = 0
            1./9., 1./9., 1./9., 1./9.,     // i = 1, 2, 3, 4
            1./36., 1./36., 1./36., 1./36., // i = 5, 6, 7, 8 
        };

        float uref; // physical reference velocity scale
        const float urefL = .2; // LBM reference velocity

        float &dx = D; // to keep notation same as the paper, reference variable to D (D notation from Jo Stam)
        const float dxL = 1.; // LBM delta x

        float fps; // desired framerate of simulation
        const float dtL = 1.; // LBM delta t
        float T; // current simulation time
        unsigned int frame; // number of frames completed

        float viscL; // lattice viscosity
        float cs; // lattice speed of sound
        float tau; // relaxation time in lattice units
        
        // calculates initial distribution values and sets up grid
        // dummy_dt because dt is calculated
        void initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dummy_dt);

        bool calcNextStep( );

        void addForce(Vector_ &dst, const Vector_ &src, const Vector_ &force, float dt);

        // calcs fbar for the ith velocity at the grid position (j,k)
        float calc_fbari(int i, int j, int k);

        // calc Fi approximation
        float calc_Fi(int i, int j, int k);

        // adjusts uref according to umax to maintain stability
        // CURRENTLY NOT SUPPORTED: DONT USE
        // ISSUE: Changing uref changes the lattice speed of sound
        // need to update velocity discretization accordingly (c = sqrt(3)*cs)
        // The streaming step will need to instead be
        void adj_uref();

        // calcs rho and U fields
        void calcPhysicalVals();

        void calcPhysicalVals(int j, int k);

};

} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/LBMSolver.cpp>
#endif

#endif