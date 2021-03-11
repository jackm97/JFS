#ifndef LBMSOLVER_H
#define LBMSOLVER_H

#include "jfs_inline.h"

#include <jfs/base/fluidBase.h>
#include <jfs/base/grid2D.h>

#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace jfs {

struct PressureWave;

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
        
        LBMSolver(unsigned int N, float L, BoundType btype, int iter_per_frame, float rho0=1.3, float visc = 1e-4, float uref = 1);

        void initialize(unsigned int N, float L, BoundType btype, int iter_per_frame, float rho0=1.3, float visc = 1e-4, float uref = 1);

        void resetFluid();

        void getImage(Eigen::VectorXf &img);

        bool calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources);

        bool calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources, const std::vector<PressureWave> p_waves);

        // density visualization

        void setDensityVisBounds(float minrho, float maxrho);

        void getCurrentDensityBounds(float minmax_rho[2]);

        void enableDensityViewMode(bool use);

        ~LBMSolver(){};

    private:

        bool view_density_ = false;

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

        int iter_per_frame;

        float uref; // physical reference velocity scale
        const float urefL = .2; // LBM reference velocity

        float &dx = D; // to keep notation same as the paper, reference variable to D (D notation from Jo Stam)
        const float dxL = 1.; // LBM delta x

        float fps; // desired framerate of simulation
        const float dtL = 1.; // LBM delta t
        float T; // current simulation time

        float viscL; // lattice viscosity
        float cs; // lattice speed of sound
        float tau; // relaxation time in lattice units

        float minrho;
        float maxrho;

    private:

        bool calcNextStep( std::vector<PressureWave> p_waves );

        void addForce(Vector_ &dst, const Vector_ &src, const Vector_ &force, float dt);

        // calcs fbar for the ith velocity at the grid position (j,k)
        float calc_fbari(int i, int j, int k);

        // calc Fi approximation
        float calc_Fi(int i, int j, int k);

        // adjusts uref according to umax to maintain stability
        // CURRENTLY NOT SUPPORTED: DONT USE
        // ISSUE: Changing uref changes the lattice speed of sound
        // need to update velocity discretization accordingly (c = sqrt(3)*cs)
        void adj_uref();

        // calcs rho and U fields
        void calcPhysicalVals();

        void calcPhysicalVals(int j, int k);

        void forceVelocity(int i, int j, float ux, float uy);

        void doPressureWave(PressureWave p_wave);

        void doBoundaryDamping();

        void getDensityImage(Eigen::VectorXf &image);

        void getSourceImage(Eigen::VectorXf &image);

};

struct PressureWave
{
    Eigen::Vector3f x = Eigen::Vector3f::Zero(); // center position
    
    Eigen::Vector3f u = Eigen::Vector3f::Zero(); // center speed

    float u_imp = 0; // peak fluid speed

    float radius = 0; // pressure wave

    float t_start = 0; // start time

    bool skadoosh = false;
};

} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/LBMSolver.cpp>
#endif

#endif