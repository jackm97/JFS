#ifndef LBMSOLVER_CUDA_H
#define LBMSOLVER_CUDA_H

#include "../jfs_inline.h"

#include <jfs/cuda/grid/cuda_grid2d.h>

#include <vector>

#ifndef __CUDA_ARCH__
    #define __HOST__DEVICE__
    #define __HOST__
    #define __DEVICE__
#else
    #define __HOST__DEVICE__ __host__ __device__
    #define __HOST__ __host__
    #define __DEVICE__ __device__
#endif

namespace jfs {

// struct to pass solver properties to
// kernel that creates gpu copy of
// the solver
struct LBMSolverProps
{
    int N;
    float L;
    BoundType btype;
    int iter_per_frame;
    float rho0;
    float visc;
    float uref;

    float* rho;
    float* rho_mapped;

    float* f;
    float* f0;

    float* U;
    float* F;
};

class cudaLBMSolver : public cudaGrid2D {

    public:
        // constructors
        __HOST__DEVICE__
        cudaLBMSolver(){};
        
        __HOST__
        cudaLBMSolver(unsigned int N, float L, BoundType btype, int iter_per_frame, float rho0=1.3, float visc = 1e-4, float uref = 1);

        // initializer
        __HOST__DEVICE__
        void initialize(unsigned int N, float L, BoundType btype, int iter_per_frame, float rho0=1.3, float visc = 1e-4, float uref = 1);

        // reset simulation data
        __HOST__
        void resetFluid();

        // do next simulation steps
        __HOST__
        bool calcNextStep(const std::vector<Force> forces);

        // apply force to reach velocity
        __HOST__DEVICE__
        void forceVelocity(int i, int j, float ux, float uy);

        // density mapping
        __HOST__
        void setDensityMapping(float minrho, float maxrho);

        __HOST__
        void densityExtrema(float minmax_rho[2]);

        // inline getters:
        __HOST__DEVICE__
        float TimeStep(){return this->dt;}

        __HOST__DEVICE__
        float Time(){return this->T;}

        __HOST__DEVICE__
        float DeltaX(){return this->dx;}

        __HOST__DEVICE__
        float soundSpeed(){return this->us;}

        __HOST__DEVICE__
        float Rho0(){return this->rho0;}

        // non-inline getters
        __HOST__
        float* rhoData();

        __HOST__
        float* mappedRhoData();

        __HOST__
        float* velocityData();

        //destructor
        __HOST__DEVICE__
        ~cudaLBMSolver();

    #ifndef __CUDA_ARCH__
    private:
    #else
    public:
    #endif

        // DEV NOTES:
        // The paper uses fi to represent the discretized distribution function
        // Because of this, j,k,l are now used to index x,y,z positions on the grid

        bool is_initialized_ = false;

        float* f; // the distribution function
        float* f0; // the distribution function

        float* rho_; // calculated rho from distribution function
        float* cpu_rho_; // calculated rho from distribution function
        float minrho_; 
        float maxrho_;
        float* rho_mapped_; // rho_, but mapped to [0,1] with min/maxrho_
        float* cpu_rho_mapped_ ;

        float* U;
        float* cpu_U_ ;

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

        int iter_per_frame;

       
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

        void** gpu_this_ptr; // shallow copy of this pointer on gpu

    #ifndef __CUDA_ARCH__
    private:
    #else
    public:
    #endif
        __HOST__
        bool calcNextStep();

        // calc Fi approximation
        __DEVICE__
        float calc_Fi(int i, int j, int k);

        __DEVICE__
        float calc_fbari(int i, int j, int k);

        // calcs rho and U fields
        // __HOST__DEVICE__
        // void calcPhysicalVals();

        __DEVICE__
        void calcPhysicalVals(int j, int k);

        __HOST__
        void mapDensity(); // used by cuda kernel, must be public

        __HOST__
        void doBoundaryDamping();

        __HOST__DEVICE__
        void clearGrid();
};

} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/cuda/lbm_solver_cuda.cu>
#endif

#endif