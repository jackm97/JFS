#ifndef LBMSOLVER_H
#define LBMSOLVER_H

#include "jfs_inline.h"

#include <jfs/base/grid2D.h>

#include <vector>

namespace jfs {

class LBMSolver : public grid2D {

    public:
        // constructors
        LBMSolver(){};
        
        LBMSolver(unsigned int N, float L, BoundType btype, float rho0=1.3, float visc = 1e-4, float uref = 1);

        // initializer
        void Initialize(unsigned int N, float L, BoundType btype, float rho0= 1.3, float visc = 1e-4, float uref = 1);

        // reset simulation data
        void ResetFluid();

        // do next simulation step
        bool CalcNextStep(const std::vector<Force> forces);

        // apply force to reach velocity
        void ForceVelocity(int i, int j, float ux, float uy);

        // density mapping
        void SetDensityMapping(float minrho, float maxrho);

        void DensityExtrema(float *minmax_rho);

        // inline getters:
        float TimeStep(){return this->dt;}

        float Time(){return this->T;}

        float DeltaX(){return this->dx;}

        float SoundSpeed(){return this->us;}

        float Rho0(){return this->rho0;}

        float* RhoData(){return this->rho_;}

        float* MappedRhoData(){ MapDensity(); return this->rho_mapped_;}

        float* VelocityData(){return this->U;}

        //destructor
        ~LBMSolver(){ ClearGrid(); };

    private:

        // DEV NOTES:
        // The paper uses fi to represent the discretized distribution function
        // Because of this, j,k,l are now used to index x,y,z positions on the grid

        bool is_initialized_ = false;

        float* f; // the distribution function
        float* f0; // the distribution function

        float* rho_; // calculated rho from distribution function
        float minrho_; 
        float maxrho_;
        float* rho_mapped_; // rho_, but mapped to [0,1] with min/maxrho_

        float* U; 

        float* F;

        const float c[9][2]{ // D2Q9 velocity dicretization
            {0,0},                                // i = 0
            {1,0}, {-1,0}, {0,1}, {0,-1},   // i = 1, 2, 3, 4
            {1,1}, {-1,1}, {1,-1}, {-1,-1}  // i = 5, 6, 7, 8
        };

        const float bounce_back_indices_[9]{
            0,
            2, 1, 4, 3,
            8, 7, 6, 5
        };

        const float w[9] = { // lattice weights
            4./9.,                          // i = 0
            1./9., 1./9., 1./9., 1./9.,     // i = 1, 2, 3, 4
            1./36., 1./36., 1./36., 1./36., // i = 5, 6, 7, 8 
        };

       
        float rho0; // typical physical density of fluid

        float uref; // physical reference velocity scale
        const float urefL = .2; // LBM reference velocity
        float us;  // speed of sound of fluid
        float cs; // lattice speed of sound

        float &dx = D; // to keep notation same as the paper, reference variable to D (D notation from Jo Stam)
        const float dxL = 1.; // LBM delta x

        float dt; // physical time step
        const float dtL = 1.; // LBM delta t
        float T; // current simulation time
        
        float visc; // fluid viscosity
        float viscL; // lattice viscosity
        float tau; // relaxation time in lattice units

    private:

        void ClearGrid();

        bool calcNextStep();

        // calcs fbar for the ith velocity at the grid position (j,k)
        float CalcEquilibriumDistribution(int i, int j, int k);

        // calc Fi approximation
        float CalcLatticeForce(int i, int j, int k);

        // calcs rho and U fields
        void CalcPhysicalVals();

        void CalcPhysicalVals(int j, int k);

        void DoBoundaryDamping();

        void MapDensity();
};

} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/LBMSolver.cpp>
#endif

#endif