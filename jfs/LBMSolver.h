#ifndef LBMSOLVER_H
#define LBMSOLVER_H

#include "jfs_inline.h"

#include <jfs/base/fluid2D.h>

#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace jfs {

class LBMSolver : public fluid2D {

    public:
        // typical density of fluid
        float rho0;
        // fluid viscosity
        float visc; 
        // speed of sound of fluid
        float us;  

        LBMSolver(){};
        
        LBMSolver(unsigned int N, float L, float fps, float rho0=1.3, float visc=.01, float us=1024.);

        void initialize(unsigned int N, float L, float fps, float rho0=1.3, float visc=.01, float us=1024.);

        void resetFluid();

        void calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources);

        ~LBMSolver(){};

    private:
        // DEV NOTES:
        // The paper uses fi to represent the discretized distribution function
        // Because of this, j,k,l are now used to index x,y,z positions on the grid

        Eigen::VectorXf &f = U0; // U0 is unused for this solver, to save space it is reassigned to f (the distribution function)
        Eigen::VectorXf rho; // calculated rho from distribution function

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

        Eigen::VectorXf X0; // position values of particle at X at time t - dt ( used for advecting sources )

        Eigen::SparseVector<float> Fb; // boundary forces;
        Eigen::VectorXf Ub; // velcoity field after bounds are applied

        float uref; // physical reference velocity scale
        const float urefL = .5; // LBM reference velocity

        float &dx = D; // to keep notation same as the paper, reference variable to D (D notation from Jo Stam)
        const float dxL = 1.; // LBM delta x

        float fps; // desired framerate of simulation
        float dt = 0.; // accumulated physical dt, reset to 0 once dt = 1/fps
        const float dtL = 1.; // LBM delta t
        
        // calculates initial distribution values and sets up grid
        void initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dt);

        // calls initializeGridProperties and setXGrid
        void initializeGrid(unsigned int N, float L, BOUND_TYPE BOUND, float dt);

        void calcNextStep( );

        void addForce(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &force, float dt);

        // calcs fbar for the ith velocity at the grid position (j,k)
        float calc_fbari(int i, int j, int k);

        // calc Fi approximation
        float calc_Fi(int i, int j, int k);

        // adjusts uref according to umax to maintain stability
        void adj_uref();

        // calcs rho and U fields
        void calcPhysicalVals();

        // boundary forces 
        void addBoundaryForces(Eigen::VectorXf &Ub, Eigen::VectorXf &U, float dt);

};

} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/LBMSolver.cpp>
#endif

#endif