#include "lbm_cuda_kernels.h"

#include <jfs/cuda/grid/cuda_grid2d.h>
#include <cuda_runtime.h>

namespace jfs {

using FieldType2D::Vector;
using FieldType2D::Scalar;

__CONSTANT__ float cs = 0.57735026919;
__CONSTANT__ float lat_uref = .2;
__CONSTANT__ int c[9][2] { // D2Q9 velocity dicretization
    {0,0},                                // i = 0
    {1,0}, {-1,0}, {0,1}, {0,-1},   // i = 1, 2, 3, 4
    {1,1}, {-1,1}, {1,-1}, {-1,-1}  // i = 5, 6, 7, 8
};

__CONSTANT__ int bounce_back_indices[9]{
    0,
    2, 1, 4, 3,
    8, 7, 6, 5
};

__CONSTANT__ float w[9] = { // lattice weights
    4./9.,                          // i = 0
    1./9., 1./9., 1./9., 1./9.,     // i = 1, 2, 3, 4
    1./36., 1./36., 1./36., 1./36., // i = 5, 6, 7, 8 
};

__CONSTANT__ LBMSolverProps const_props[1];
CudaLBMSolver* current_cuda_lbm_solver = 0x0;

/*
*
DEVICE FUNCTIONS
*
*/
// alpha represents the lattice index
__DEVICE__
float calcEquilibrium(int alpha, const float* u, float uref, float rho, float rho0)
{
    #ifdef __CUDA_ARCH__
    return w[alpha] * rho/rho0 * ( 1 + ( lat_uref/uref*c[alpha][0]*u[0] + lat_uref/uref*c[alpha][1]*u[1] )/powf(cs, 2) +
            powf(( lat_uref/uref*c[alpha][0]*u[0] + lat_uref/uref*c[alpha][1]*u[1] ), 2)/powf(cs, 4) -
            ( lat_uref/uref*u[0]*lat_uref/uref*u[0] + lat_uref/uref*u[1]*lat_uref/uref*u[1] )/(2 * powf(cs, 2)) );
    #endif
}
__DEVICE__
float calcLatticeForce(int alpha, float lat_tau, float dx, const float* force, const float* u, float uref, float rho, float rho0)
{
    #ifdef __CUDA_ARCH__
    float ci[2]{c[alpha][0], c[alpha][1]};

    float fx = force[0] * 1/rho0 * dx * powf(lat_uref/uref, 2);
    float fy = force[1] * 1/rho0 * dx * powf(lat_uref/uref, 2);

    float ci_dot_u = ci[0]*u[0]*lat_uref/uref + ci[1]*u[1]*lat_uref/uref;

    return (1 - lat_tau/2) * w[alpha] * (
         ( (1/powf(cs,2))*(ci[0] - u[0]*lat_uref/uref) + (ci_dot_u/powf(cs,4)) * ci[0] )  * fx + 
         ( (1/powf(cs,2))*(ci[1] - u[1]*lat_uref/uref) + (ci_dot_u/std::pow(cs,4)) * ci[1] )  * fy
    );
    #endif
}

__DEVICE__
void calcPhysicalProps(const float* f, float* u_data, float uref, float* rho_data, float rho0)
{
    #ifdef __CUDA_ARCH__
    u_data[0] = 0;
    u_data[1] = 0;
    rho_data[0] = 0;
    for (int alpha = 0; alpha < 9; alpha++)
    {
        printf("%.2f\n", f[alpha]);
        rho_data[0] += f[alpha];
        u_data[0] += c[alpha][0] * f[alpha];
        u_data[1] += c[alpha][1] * f[alpha];
    }
    u_data[0] = uref/lat_uref * (u_data[0] / rho_data[0]);
    u_data[1] = uref/lat_uref * (u_data[1] / rho_data[0]);
    rho_data[0] *= rho0;
    printf("%.2f\n\n", rho_data[0]);
    #endif
}
/*
*
END DEVICE FUNCTIONS
*
*/
__GLOBAL__
void forceVelocityKernel(ushort i, ushort j, float ux, float uy)
{
    
    #ifdef __CUDA_ARCH__
        CudaGrid2D<Vector> u_grid;
        u_grid.MapData(const_props[0].u_grid, const_props[0].grid_size, 1);
        CudaGrid2D<Vector> force_grid;
        force_grid.MapData(const_props[0].force_grid, const_props[0].grid_size, 1);
        CudaGrid2D<Scalar> rho_grid;
        rho_grid.MapData(const_props[0].rho_grid, const_props[0].grid_size, 1);

        force_grid(i, j, 0, 0) += (ux - u_grid(i, j, 0, 0)) * rho_grid(i, j, 0, 0) / const_props[0].dt;
        force_grid(i, j, 0, 1) += (ux - u_grid(i, j, 0, 1)) * rho_grid(i, j, 0, 1) / const_props[0].dt;
    #endif
}

__GLOBAL__
void resetDistributionKernel(float* f_data)
{

    int grid_size = const_props[0].grid_size;
    int alpha = blockIdx.x * blockDim.x + threadIdx.x;
    int j = alpha/(grid_size*9);
    alpha -= grid_size*9*j;
    int i = alpha/(9);
    alpha -= 9*i;

    #ifdef __CUDA_ARCH__

        if (i >= grid_size || j >= grid_size)
            return;
            
        float* f = f_data + grid_size*9*j + 9*i;
        
        const float* u_ptr = const_props[0].u_grid + grid_size*2*j + 2*i;
        
        float rho = *(const_props[0].rho_grid + grid_size*1*j + 1*i);
        
        f[alpha] = calcEquilibrium(alpha, u_ptr, const_props[0].uref, rho, const_props[0].rho0);
    #endif
}

__GLOBAL__
void collideKernel()
{

    int grid_size = const_props[0].grid_size;
    int alpha = blockIdx.x * blockDim.x + threadIdx.x;
    int j = alpha/(grid_size*9);
    alpha -= grid_size*9*j;
    int i = alpha/(9);
    alpha -= 9*i;

    #ifdef __CUDA_ARCH__

        if (i >= grid_size || j >= grid_size)
            return;
        
        float* f = const_props[0].f_grid + grid_size*9*j + 9*i;
        
        float* fbar = const_props[0].f0_grid + grid_size*9*j + 9*i;
        
        float rho =  *(const_props[0].rho_grid + grid_size*1*j + 1*i);
        
        const float* u_ptr = const_props[0].u_grid + grid_size*2*j + 2*i;
        
        const float* force_ptr = const_props[0].force_grid + grid_size*2*j + 2*i;

        float lat_force;

        lat_force = calcLatticeForce(alpha, const_props[0].lat_tau, const_props[0].dt, force_ptr, u_ptr, const_props[0].uref, rho, const_props[0].rho0);
        f[alpha] += lat_force - (f[alpha] - fbar[alpha])/const_props[0].lat_tau;
        fbar[alpha] = f[alpha];
    #endif
}

__GLOBAL__
void streamKernel()
{

    int grid_size = const_props[0].grid_size;
    int alpha = blockIdx.x * blockDim.x + threadIdx.x;
    int j = alpha/(grid_size*9);
    alpha -= grid_size*9*j;
    int i = alpha/(9);
    alpha -= 9*i;

    #ifdef __CUDA_ARCH__

        if (i >= grid_size || j >= grid_size)
            return;

        int cix = c[alpha][0];
        int ciy = c[alpha][1];
        
        float* f = const_props[0].f_grid + grid_size*9*j + 9*i;

        if ((j-ciy) >= 0 && (j-ciy) < grid_size && (i-cix) >= 0 && (i-cix) < grid_size)
            f[alpha] = const_props[0].f0_grid[grid_size*9*(j-ciy) + 9*(i-cix) + alpha];
        else
        {
            int alpha_bounce = bounce_back_indices[alpha];
            f[alpha] = const_props[0].f0_grid[grid_size*9*j + 9*i + alpha_bounce];
        }
    #endif
}

__GLOBAL__
void calcPhysicalKernel()
{

    int grid_size = const_props[0].grid_size;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = i/(grid_size);
    i -= grid_size*j;

    #ifdef __CUDA_ARCH__

        if (i >= grid_size || j >= grid_size)
            return;
    
        float* f = const_props[0].f_grid + grid_size*9*j + 9*i;
        float* u_data = const_props[0].u_grid + grid_size*2*j + 2*i;
        float* rho_data = const_props[0].rho_grid + grid_size*1*j + 1*i;
        u_data[0] = 0;
        u_data[1] = 0;
        rho_data[0] = 0;
        for (int alpha = 0; alpha < 9; alpha++)
        {
            rho_data[0] += f[alpha];
            u_data[0] += c[alpha][0] * f[alpha];
            u_data[1] += c[alpha][1] * f[alpha];
        }
        u_data[0] = const_props[0].uref/lat_uref * (u_data[0] / rho_data[0]);
        u_data[1] = const_props[0].uref/lat_uref * (u_data[1] / rho_data[0]);
        rho_data[0] *= const_props[0].rho0;
    #endif
}

__GLOBAL__
void boundaryDampKernel()
{

    int grid_size = const_props[0].grid_size;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = i/(grid_size);
    i -= grid_size*j;

    #ifdef __CUDA_ARCH__

        if (i != (grid_size-1) && i != (0) && j != (grid_size-1) && j != (0))
            return;

        if (i == (grid_size-1) || i == (0))
        {
            int step;
            if (i == 0)
                step = 1;
            else
                step = -1;
            
            i += step;

            const_props[0].u_grid[grid_size*2*j + 2*(i-step) + 0] = const_props[0].u_grid[grid_size*2*j + 2*i + 0];
            const_props[0].u_grid[grid_size*2*j + 2*(i-step) + 1] = const_props[0].u_grid[grid_size*2*j + 2*i + 1];
            const_props[0].rho_grid[grid_size*2*j + 2*(i-step)] = const_props[0].rho_grid[grid_size*2*j + 2*i];
        }
        else
        {
            int step;
            if (j == 0)
                step = 1;
            else
                step = -1;
            
            j += step;

            const_props[0].u_grid[grid_size*2*(j-step) + 2*i + 0] = const_props[0].u_grid[grid_size*2*j + 2*i + 0];
            const_props[0].u_grid[grid_size*2*(j-step) + 2*i + 1] = const_props[0].u_grid[grid_size*2*j + 2*i + 1];
            const_props[0].rho_grid[grid_size*2*(j-step) + 2*i] = const_props[0].rho_grid[grid_size*2*j + 2*i];
        } 
    #endif
}

} // namespace jfs