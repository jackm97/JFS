#ifndef LBM_CUDA_KERNELS_H
#define LBM_CUDA_KERNELS_H

#include <jfs/cuda/lbm_solver_cuda.h>

namespace jfs {

    __global__
    void forceVelocityKernel(int i, int j, float ux, float uy);

    __global__
    void addMassKernel(int i, int j, float rho);

    __global__
    void resetDistributionKernel(float *f_data);

    __global__
    void collideKernel(bool *flag_ptr);

    __global__
    void streamKernel(bool *flag_ptr);

    __global__
    void calcRhoKernel();

    __global__
    void calcVelocityKernel();

    __global__
    void boundaryDampKernel();

} // namespace jfs

#endif