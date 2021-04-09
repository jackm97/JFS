#include "lbm_solver_cuda.h"

#include <jfs/cuda/lbm_cuda_kernels.h>
#include <cuda_runtime.h>

#include <cstring>
#include <iostream>

namespace jfs {

JFS_INLINE CudaLBMSolver::CudaLBMSolver(ushort grid_size, float grid_length, BoundType btype, float rho0, float visc, float uref) :
cs_{ 1/sqrtf(3) }
{
    Initialize(grid_size, grid_length, btype, rho0, visc, uref);
}

JFS_INLINE void CudaLBMSolver::Initialize(ushort grid_size, float grid_length, BoundType btype, float rho0, float visc, float uref)
{
    grid_size_ = grid_size;
    grid_length_ = grid_length;
    btype_ = btype;

    rho0_ = rho0;
    visc_ = visc;
    uref_ = uref;
    
    // lattice scaling stuff
    us_ = cs_/lat_uref_ * uref_;
    
    dx_ = grid_length_ / (float)grid_size_;
    lat_visc_ = lat_uref_/(uref_ * dx_) * visc;
    lat_tau_ = (3*lat_visc_ + .5);
    dt_ = lat_uref_/uref_ * dx_ * lat_dt_;

    f_grid_.Resize(grid_size_, 9);
    f0_grid_.Resize(grid_size_, 9);

    rho_grid_.Resize(grid_size_, 1);
    rho_grid_mapped_.Resize(grid_size_, 3);

    u_grid_.Resize(grid_size_, 1);

    force_grid_.Resize(grid_size_, 1);

    LBMSolverProps props = SolverProps();
    cudaMemcpyToSymbol(const_props, &props, sizeof(LBMSolverProps), 0, cudaMemcpyHostToDevice);
    current_cuda_lbm_solver = this;
    cudaDeviceSynchronize();

    ResetFluid();
}

JFS_INLINE void CudaLBMSolver::ResetFluid()
{
    for (int d = 0; d < 2; d++)
    {
        u_grid_.SetGridToValue(0, 0, d);
        force_grid_.SetGridToValue(0, 0, d);
    }
    rho_grid_.SetGridToValue(rho0_, 0, 0);

    int threads_per_block = 256;
    int num_blocks = (9*(int)grid_size_*(int)grid_size_) / threads_per_block + 1;
    resetDistributionKernel KERNEL_ARG2(num_blocks, threads_per_block) (f_grid_.Data());
    cudaDeviceSynchronize();

    time_ = 0;
}

JFS_INLINE bool CudaLBMSolver::CalcNextStep(const std::vector<Force> forces)
{
    bool failedStep = false;
    try
    {   
        for (int i = 0; i < forces.size(); i++)
        {
            float force[3] = {
                forces[i].force[0],
                forces[i].force[1],
                forces[i].force[2]
            };
            float point[3] = {
                forces[i].pos[0]/grid_length_ * grid_size_,
                forces[i].pos[1]/grid_length_ * grid_size_,
                forces[i].pos[2]/grid_length_ * grid_size_
            };
            if (point[0] < grid_size_ && point[0] >= 0 && point[1] < grid_size_ && point[1] >= 0)
                for (int d = 0; d < 2; d++)
                {
                    force_grid_.InterpToGrid(force[d], point[0], point[1], 0, d);
                }
        }
        
        failedStep = CalcNextStep();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        failedStep = true;
    }
    for (int d = 0; d < 2; d++)
    {
        force_grid_.SetGridToValue(0, 0, d);
    }

    if (failedStep) ResetFluid();

    return failedStep;
}

JFS_INLINE void CudaLBMSolver::ForceVelocity(ushort i, ushort j, float ux, float uy)
{
    forceVelocityKernel KERNEL_ARG2(1, 1) (i, j, ux, uy);
    cudaDeviceSynchronize();
}

JFS_INLINE void CudaLBMSolver::SetDensityMapping(float minrho, float maxrho)
{
    minrho_ = minrho;
    maxrho_ = maxrho;
}

JFS_INLINE void CudaLBMSolver::DensityExtrema(float minmax_rho[2])
{
    float* rho_grid_host = rho_grid_.HostData();

    float minrho = rho_grid_host[0];
    float maxrho = rho_grid_host[0];

    for (int i=0; i < grid_size_*grid_size_; i++)
    {
        if (rho_grid_host[i] < minrho)
            minrho = rho_grid_host[i];
    }

    for (int i=0; i < grid_size_*grid_size_; i++)
    {
        if (rho_grid_host[i] > maxrho)
            maxrho = rho_grid_host[i];
    }

    minmax_rho[0] = minrho;
    minmax_rho[1] = maxrho;
}

JFS_INLINE bool CudaLBMSolver::CalcNextStep()
{
    if (current_cuda_lbm_solver != this)
    {
        LBMSolverProps props = SolverProps();
        cudaMemcpyToSymbol(const_props, &props, sizeof(LBMSolverProps), 0, cudaMemcpyHostToDevice);
        current_cuda_lbm_solver = this;
    }
    bool failed_step = false;

    int threads_per_block = 256;
    int num_blocks = (9*(int)grid_size_*(int)grid_size_) / threads_per_block + 1;
    
    resetDistributionKernel KERNEL_ARG2(num_blocks, threads_per_block) (f0_grid_.Data());
    collideKernel KERNEL_ARG2(num_blocks, threads_per_block) ();
    cudaDeviceSynchronize();

    streamKernel KERNEL_ARG2(num_blocks, threads_per_block) ();
    cudaDeviceSynchronize();

    calcPhysicalKernel KERNEL_ARG2(num_blocks/9, threads_per_block) ();
    cudaDeviceSynchronize();

    // do any field manipulations before next step
    if (btype_ == DAMPED)
    {
        boundaryDampKernel KERNEL_ARG2(num_blocks/9, threads_per_block) ();
        resetDistributionKernel KERNEL_ARG2(num_blocks, threads_per_block) (f_grid_.Data());
    }

    time_ += dt_;
    
    return failed_step;
}

__host__
JFS_INLINE void CudaLBMSolver::MapDensity()
{
    float* host_rho_grid = RhoData();

    float minrho = host_rho_grid[0];
    float maxrho = host_rho_grid[0];
    float meanrho = 0;
    for (int i = 0; i < grid_size_*grid_size_; i++)
        meanrho += host_rho_grid[i];
    meanrho /= grid_size_*grid_size_;

    for (int i=0; i < grid_size_*grid_size_ && minrho_ == -1; i++)
    {
        if (host_rho_grid[i] < minrho)
            minrho = host_rho_grid[i];
    }

    if (minrho_ != -1)
        minrho = minrho_;

    for (int i=0; i < grid_size_*grid_size_ && maxrho_ == -1; i++)
    {
        if (host_rho_grid[i] > maxrho)
            maxrho = host_rho_grid[i];
    }

    if (maxrho_ == -1 && minrho_ == -1)
    {
        if (maxrho - meanrho > meanrho - minrho)
            minrho = meanrho - (maxrho - meanrho);
        else
            maxrho = meanrho - (minrho - meanrho);
    }

    if (maxrho_ != -1)
        maxrho = maxrho_;


    float* rho_grid_mapped_host = rho_grid_mapped_.HostData();
    for (int i=0; i < grid_size_; i++)
        for (int j=0; j < grid_size_; j++)
        {
            float rho;
            rho = host_rho_grid[grid_size_*j + i];
            if ((maxrho - minrho) != 0)
                rho = (rho- minrho)/(maxrho - minrho);
            else
                rho = 0 * rho;

            // rho = (rho < 0) ? 0 : rho;
            // rho = (rho > 1) ? 1 : rho;

            rho_grid_mapped_host[grid_size_*3*j + 3*i + 0] = rho;
            rho_grid_mapped_host[grid_size_*3*j + 3*i + 1] = rho;
            rho_grid_mapped_host[grid_size_*3*j + 3*i + 2] = rho;
        }

    cudaMemcpy(rho_grid_mapped_.Data(), rho_grid_mapped_host, 3*grid_size_*grid_size_*sizeof(float), cudaMemcpyHostToDevice);
}

JFS_INLINE LBMSolverProps CudaLBMSolver::SolverProps()
{

    LBMSolverProps props;
    props.grid_size = grid_size_;
    props.grid_length = grid_length_;
    props.btype = btype_;
    props.rho0 = rho0_;
    props.visc = visc_;
    props.lat_visc = lat_visc_;
    props.lat_tau = lat_tau_;
    props.uref = uref_;
    props.dt = dt_;
    props.rho_grid = rho_grid_.Data();
    props.f_grid = f_grid_.Data();
    props.f0_grid = f0_grid_.Data();
    props.u_grid = u_grid_.Data();
    props.force_grid = force_grid_.Data();
    props.failed_step = false;

    return props;
}

} // namespace jfs