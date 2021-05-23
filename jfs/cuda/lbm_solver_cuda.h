#ifndef LBMSOLVER_CUDA_H
#define LBMSOLVER_CUDA_H

#include "../jfs_inline.h"

#include <jfs/cuda/grid/cuda_grid2d.h>

#include <vector>
#include <cmath>

namespace jfs {

// struct to pass solver properties to
// kernel that creates gpu copy of
// the solver
    struct LBMSolverProps {
        uint grid_size;
        float grid_length;
        BoundType btype;

        float rho0;
        float lat_tau;
        float uref;
        float dt;

        float *rho_grid;

        float *f_grid;
        float *f0_grid;

        float *u_grid;
        float *force_grid;
    };

    class CudaLBMSolver {

    public:
        // constructors
        CudaLBMSolver() : cs_{1 / sqrtf(3)} {}

        CudaLBMSolver(uint grid_size, float grid_length, BoundType btype, float rho0 = 1.3, float visc = 1e-4,
                      float uref = 1);

        // initializer
        void Initialize(uint grid_size, float grid_length, BoundType btype, float rho0 = 1.3, float visc = 1e-4,
                        float uref = 1);

        // reset simulation data
        void ResetFluid();

        // do next simulation steps
        bool CalcNextStep(const std::vector<Force> &forces);

        // apply force to reach velocity
        void ForceVelocity(int *i, int *j, float *ux, float *uy, int num_points);

        void AddMassSource(int i, int j, float rho);

        // density mapping

        void SetDensityMapping(float min_rho, float max_rho);

        void DensityExtrema(float minmax_rho[2]);

        void RegisterRhoMapTexture(uint tex_id);

        void MapRhoData2Texture();

        // inline getters:

        float TimeStep() { return dt_; }

        float Time() { return time_; }

        float DeltaX() { return dx_; }

        float SoundSpeed() { return us_; }

        float Rho0() { return rho0_; }

        float Viscosity() { return visc_; }

        float *RhoData() {SyncHostWithDevice(); return rho_grid_.HostData(); }

        float *MappedRhoData() {
            MapDensity();
            mapped_rho_grid_.SyncHostWithDevice();
            return mapped_rho_grid_.HostData();
        }

        float *VelocityData() {SyncHostWithDevice(); return u_grid_.HostData(); }

        // inline indexers:

        float IndexRhoData(int i, int j) {return rho_grid_.Index(i, j, 0, 0); }

        float IndexVelocityData(int i, int j, int d) {return u_grid_.Index(i, j, 0, d); }

        //destructor
        ~CudaLBMSolver();

    private:

        // grid stuff
        BoundType btype_;
        uint grid_size_{};
        float grid_length_{};

        CudaGrid2D<FieldType2D::Scalar> f_grid_; // the distribution function
        CudaGrid2D<FieldType2D::Scalar> f0_grid_; // the distribution function

        CudaGrid2D<FieldType2D::Scalar> rho_grid_; // calculated rho from distribution function
        float min_rho_map_{};
        float max_rho_map_{};
        CudaGrid2D<FieldType2D::Scalar> mapped_rho_grid_; // rho_, but mapped to [0,1] with min/max_rho_map_

        CudaGrid2D<FieldType2D::Vector> u_grid_;

        CudaGrid2D<FieldType2D::Vector> force_grid_;

        float rho0_{}; // typical physical density of fluid

        float uref_{}; // physical reference velocity scale
        const float lat_uref_ = .2; // LBM reference velocity
        float us_{};  // speed of sound of fluid
        const float cs_; // lattice speed of sound

        float dx_{}; // to keep notation same as the paper, reference variable to D (D notation from Jo Stam)
        const float lat_dx_ = 1.; // LBM delta x

        float dt_{}; // physical time step
        const float lat_dt_ = 1.; // LBM delta t
        float time_{}; // current simulation time

        float visc_{}; // fluid viscosity
        float lat_tau_{}; // relaxation time in lattice units

        bool host_synced_ = false;

        // opengl interop
        // note that they are all void* so that this class can be included
        // in a .cpp file without throwing errors
        void* tex_resource_; // cast to (cudaGraphicsResource_t)
        void* tex_array_; // cast to (cudaArray_t)
        void* tex_surf_; // cast to (cudaSurfaceObject_t)
        bool is_resource_registered_ = false;

    private:
        bool CalcNextStep();

        void SyncHostWithDevice(){if (host_synced_) return;
            rho_grid_.SyncHostWithDevice();
            u_grid_.SyncHostWithDevice(); host_synced_ = true;}

        void MapDensity();

        LBMSolverProps SolverProps();
    };

} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/cuda/lbm_solver_cuda.cu>
#endif

#endif