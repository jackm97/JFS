#ifndef LBM_CUDA_KERNELS_H
#define LBM_CUDA_KERNELS_H

#include <jfs/cuda/lbm_solver_cuda.h>

#ifdef __INTELLISENSE__
void __syncthreads();  // workaround __syncthreads warning
#define KERNEL_ARG2(grid, block)
#define KERNEL_ARG3(grid, block, sh_mem)
#define KERNEL_ARG4(grid, block, sh_mem, stream)
#define __GLOBAL__
#define __LAUNCH_BOUNDS__(max_threads, min_blocks)
#define __CONSTANT__
#define __SHARED__
#else
#define KERNEL_ARG2(grid, block) <<< grid, block >>>
#define KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#define __GLOBAL__ __global__
#define __LAUNCH_BOUNDS__(max_threads, min_blocks) __launch_bounds__(max_threads, min_blocks)
#define __CONSTANT__ __constant__
#define __SHARED__ __shared__
#endif

namespace jfs {

extern __CONSTANT__ LBMSolverProps const_props[1];
extern CudaLBMSolver* current_cuda_lbm_solver;

__GLOBAL__
void forceVelocityKernel(ushort i, ushort j, float ux, float uy);

__GLOBAL__
void resetDistributionKernel(float* f_data);

__GLOBAL__
void collideKernel();

__GLOBAL__
void streamKernel();

__GLOBAL__
void calcPhysicalKernel();

__GLOBAL__
void boundaryDampKernel();

} // namespace jfs

#endif