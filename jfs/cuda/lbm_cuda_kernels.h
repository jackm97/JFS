#ifndef LBM_CUDA_KERNELS_H
#define LBM_CUDA_KERNELS_H

#include <jfs/cuda/lbm_solver_cuda.h>

namespace jfs {

    __global__
    void forceVelocityKernel(int *i, int *j, float *ux, float *uy, int num_points);

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

    __global__
    void getMinMaxRho(float* min_rho_ptr, float* max_rho_ptr);

    __global__
    void mapDensityKernel(float *rho_mapped_grid, float min_rho_map, float max_rho_map);

    __global__
    void mapDensity2TextureKernel(cudaSurfaceObject_t tex_array);

} // namespace jfs

#endif