#ifndef LBM_CUDA_KERNELS_H
#define LBM_CUDA_KERNELS_H

#include <jfs/cuda/lbm_solver_cuda.h>

namespace jfs {

extern __constant__ LBMSolverProps const_props[1];
extern CudaLBMSolver* current_cuda_lbm_solver;

__global__
void forceVelocityKernel(ushort i, ushort j, float ux, float uy);

__global__
void resetDistributionKernel(float* f_data);

__global__
void collideKernel();

__global__
void streamKernel();

__global__
void calcPhysicalKernel();

__global__
void boundaryDampKernel();

} // namespace jfs

#endif