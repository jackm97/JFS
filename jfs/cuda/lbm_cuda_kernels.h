#ifndef LBM_CUDA_KERNELS_H
#define LBM_CUDA_KERNELS_H

#include <jfs/cuda/lbm_solver_cuda.h>

namespace jfs {

__global__
void setupGPULBM(void** gpu_this_ptr, LBMSolverProps props);

__global__
void freeGPULBM(void** gpu_this_ptr);

__global__
void forceVelocityKernel(int i, int j, float ux, float uy, void** gpu_this_ptr);

__global__
void resetDistributionKernel(void** gpu_this_ptr);

__global__
void collideKernel(void** gpu_this_ptr, bool* flag_ptr);

__global__
void streamKernel(void** gpu_this_ptr, bool* flag_ptr);

__global__
void boundaryDampKernel(void** gpu_this_ptr);

} // namespace jfs

#endif