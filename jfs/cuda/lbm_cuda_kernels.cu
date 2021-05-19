#include "lbm_cuda_kernels.h"

namespace jfs {

    using FieldType2D::Vector;
    using FieldType2D::Scalar;

    __constant__ float cs = 0.57735026919;
    __constant__ float lat_uref = .2;
    __constant__ int c[9][2]{ // D2Q9 velocity discretization
            {0,  0},                                // i = 0
            {1,  0},
            {-1, 0},
            {0,  1},
            {0,  -1},   // i = 1, 2, 3, 4
            {1,  1},
            {-1, 1},
            {1,  -1},
            {-1, -1}  // i = 5, 6, 7, 8
    };

    __constant__ int bounce_back_indices[9]{
            0,
            2, 1, 4, 3,
            8, 7, 6, 5
    };

    __constant__ float w[9] = { // lattice weights
            4. / 9.,                          // i = 0
            1. / 9., 1. / 9., 1. / 9., 1. / 9.,     // i = 1, 2, 3, 4
            1. / 36., 1. / 36., 1. / 36., 1. / 36., // i = 5, 6, 7, 8
    };

    __constant__ LBMSolverProps const_props[1]{};
    CudaLBMSolver *current_cuda_lbm_solver = nullptr;

/*
*
DEVICE FUNCTIONS
*
*/
// alpha represents the lattice index
    __device__
    float calcEquilibrium(int alpha, int i, int j) {
#ifdef __CUDA_ARCH__
        float ci[2]{(float) c[alpha][0], (float) c[alpha][1]};

        const float &rho = *(const_props[0].rho_grid + const_props[0].grid_size * 1 * j + 1 * i);

        const float *u = const_props[0].u_grid + const_props[0].grid_size * 2 * j + 2 * i;

        float &rho0 = const_props[0].rho0;
        float &uref = const_props[0].uref;

        float u_cpy[2]{u[0], u[1]};
        u_cpy[0] *= lat_uref / uref;
        u_cpy[1] *= lat_uref / uref;

        float ci_dot_u = ci[0] * u_cpy[0] + ci[1] * u_cpy[1];
        float u_dot_u = u_cpy[0] * u_cpy[0] + u_cpy[1] * u_cpy[1];

        return w[alpha] * rho / rho0 *
               (1 + ci_dot_u / (powf(1 / sqrtf(3), 2)) + powf(ci_dot_u, 2) / (2 * powf(1 / sqrtf(3), 4)) -
                u_dot_u / (2 * powf(1 / sqrtf(3), 2)));
#endif
    }

    __device__
    float calcLatticeForce(int alpha, int i, int j) {
#ifdef __CUDA_ARCH__
        const float ci[2]{(float) c[alpha][0], (float) c[alpha][1]};

        const float *u = const_props[0].u_grid + const_props[0].grid_size * 2 * j + 2 * i;

        const float *force = const_props[0].force_grid + const_props[0].grid_size * 2 * j + 2 * i;

        const float &rho0 = const_props[0].rho0;
        const float dx = const_props[0].grid_length / ((float) const_props[0].grid_size - 1.f);
        const float &uref = const_props[0].uref;

        float force_cpy[2]{force[0], force[1]};
        force_cpy[0] *= 1 / rho0 * dx * powf(lat_uref / uref, 2);
        force_cpy[1] *= 1 / rho0 * dx * powf(lat_uref / uref, 2);

        float u_cpy[2]{u[0], u[1]};
        u_cpy[0] *= lat_uref / uref;
        u_cpy[1] *= lat_uref / uref;

        float ci_dot_u = ci[0] * u_cpy[0] * +ci[1] * u_cpy[1];

        return (1 - const_props[0].lat_tau / 2) * w[alpha] * (
                ((1 / powf(cs, 2)) * (ci[0] - u_cpy[0]) + (ci_dot_u / powf(cs, 4)) * ci[0]) * force_cpy[0] +
                ((1 / powf(cs, 2)) * (ci[1] - u_cpy[1]) + (ci_dot_u / powf(cs, 4)) * ci[1]) * force_cpy[1]
        );
#endif
    }

    __device__
    void calcPhysicalProps(int i, int j) {
#ifdef __CUDA_ARCH__

        const float *f = const_props[0].f_grid + const_props[0].grid_size * 9 * j + 9 * i;

        float &rho = *(const_props[0].rho_grid + const_props[0].grid_size * 1 * j + 1 * i);

        float *u = const_props[0].u_grid + const_props[0].grid_size * 2 * j + 2 * i;

        float &rho0 = const_props[0].rho0;
        float &uref = const_props[0].uref;

        u[0] = 0;
        u[1] = 0;
        rho = 0;
        for (int alpha = 0; alpha < 9; alpha++) {
            rho += f[alpha];
            u[0] += (float) c[alpha][0] * f[alpha];
            u[1] += (float) c[alpha][1] * f[alpha];
        }
        u[0] = uref / lat_uref * (u[0] / rho);
        u[1] = uref / lat_uref * (u[1] / rho);
        rho *= rho0;
#endif
    }

/*
*
END DEVICE FUNCTIONS
*
*/
    __global__
    void forceVelocityKernel(int *i, int *j, float *ux, float *uy, int num_points) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef __CUDA_ARCH__
        int grid_size = const_props[0].grid_size;
        float &ux_old = *(const_props[0].u_grid + grid_size * 2 * j[idx] + 2 * i[idx] + 0);
        float &uy_old = *(const_props[0].u_grid + grid_size * 2 * j[idx] + 2 * i[idx] + 1);
        float &force_x = *(const_props[0].force_grid + grid_size * 2 * j[idx] + 2 * i[idx] + 0);
        float &force_y = *(const_props[0].force_grid + grid_size * 2 * j[idx] + 2 * i[idx] + 1);
        float &rho = *(const_props[0].rho_grid + grid_size * j[idx] + i[idx]);

        float *f = const_props[0].f_grid + grid_size * 9 * j[idx] + 9 * i[idx];
        float *f0 = const_props[0].f0_grid + grid_size * 9 * j[idx] + 9 * i[idx];

        for (int alpha = 0; alpha < 9; alpha++) {
            float lat_force = calcLatticeForce(alpha, i[idx], j[idx]);
            float fbar = calcEquilibrium(alpha, i[idx], j[idx]);
            f[alpha] -= (lat_force - (f0[alpha] - fbar) / const_props[0].lat_tau);
        }

        ux_old -= force_x * const_props[0].dt / (2 * rho);
        uy_old -= force_y * const_props[0].dt / (2 * rho);

        force_x = (ux[idx] - ux_old) * rho / const_props[0].dt;
        force_y = (uy[idx] - uy_old) * rho / const_props[0].dt;


        ux_old += force_x * const_props[0].dt / (2 * rho);
        uy_old += force_y * const_props[0].dt / (2 * rho);

        for (int alpha = 0; alpha < 9; alpha++) {
            float lat_force = calcLatticeForce(alpha, i[idx], j[idx]);
            float fbar = calcEquilibrium(alpha, i[idx], j[idx]);
            f[alpha] += (lat_force - (f[alpha] - fbar) / const_props[0].lat_tau);
        }
#endif
    }
    __global__
    void addMassKernel(int i, int j, float rho) {

#ifdef __CUDA_ARCH__
        int grid_size = const_props[0].grid_size;

        float *f = const_props[0].f_grid + grid_size * 9 * j + 9 * i;
        rho = rho / const_props[0].rho0;

        for (int alpha = 0; alpha < 9; alpha++) {
            f[alpha] += w[alpha] * rho;
        }
#endif
    }

    __global__
    void resetDistributionKernel(float *f_data) {

        int grid_size = const_props[0].grid_size;
        int alpha = blockIdx.x * blockDim.x + threadIdx.x;
        int j = alpha / (grid_size * 9);
        alpha -= grid_size * 9 * j;
        int i = alpha / (9);
        alpha -= 9 * i;

#ifdef __CUDA_ARCH__

        if (i >= grid_size || j >= grid_size)
            return;

        float *f = f_data + grid_size * 9 * j + 9 * i;

        f[alpha] = calcEquilibrium(alpha, i, j);
#endif
    }

    __global__
    __launch_bounds__(256, 6)

    void collideKernel(bool *flag_ptr) {

        int grid_size = const_props[0].grid_size;
        int alpha = blockIdx.x * blockDim.x + threadIdx.x;
        int j = alpha / (grid_size * 9);
        alpha -= grid_size * 9 * j;
        int i = alpha / (9);
        alpha -= 9 * i;

#ifdef __CUDA_ARCH__

        if (i >= grid_size || j >= grid_size)
            return;

        float *f = const_props[0].f_grid + grid_size * 9 * j + 9 * i;
        *(const_props[0].f0_grid + grid_size * 9 * j + 9 * i + alpha) = f[alpha];
        float lat_force = calcLatticeForce(alpha, i, j);
        float fbar = calcEquilibrium(alpha, i, j);

        f[alpha] += lat_force - (f[alpha] - fbar) / const_props[0].lat_tau;

        if (isnan(f[alpha]) || isinf(f[alpha]))
            *flag_ptr = true;
#endif
    }

    __global__
    void streamKernel(bool *flag_ptr) {

        int grid_size = const_props[0].grid_size;
        int alpha = blockIdx.x * blockDim.x + threadIdx.x;
        int j = alpha / (grid_size * 9);
        alpha -= grid_size * 9 * j;
        int i = alpha / (9);
        alpha -= 9 * i;

#ifdef __CUDA_ARCH__

        if (i >= grid_size || j >= grid_size)
            return;

        int cix = c[alpha][0];
        int ciy = c[alpha][1];

        float *f = const_props[0].f_grid + grid_size * 9 * j + 9 * i;
        int alpha_bounce = bounce_back_indices[alpha];
        int i_back = (i - cix);
        int j_back = (j - ciy);
        bool is_bounce = ( i_back < 0 || i_back == grid_size || j_back < 0 || j_back == grid_size );

        float *f0 = (is_bounce) ? const_props[0].f0_grid + grid_size * 9 * j + 9 * i :
                const_props[0].f0_grid + grid_size * 9 * j_back + 9 * i_back;

        f[alpha] = (is_bounce) ? f0[alpha_bounce] : f0[alpha];

        float f_val = f[alpha];
        if (isnan(f_val) || isinf(f_val)) {
            *flag_ptr = true;
        }
#endif
    }

    __global__
    void calcRhoKernel() {

        int grid_size = const_props[0].grid_size;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = i / (grid_size);
        i -= grid_size * j;

#ifdef __CUDA_ARCH__

        if (i >= grid_size || j >= grid_size)
            return;

        const float *f = const_props[0].f_grid + grid_size * 9 * j + 9 * i;

        float rho = 0;

        float rho0 = const_props[0].rho0;

        for (int alpha = 0; alpha < 9; alpha++)
            rho += f[alpha] * rho0;

        *(const_props[0].rho_grid + grid_size * j + i) = rho;
#endif
    }

    __global__ void calcVelocityKernel() {

        int grid_size = const_props[0].grid_size;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = i / (grid_size);
        i -= grid_size * j;

#ifdef __CUDA_ARCH__

        if (i >= grid_size || j >= grid_size)
            return;

        float rho = *(const_props[0].rho_grid + grid_size * j + i);
        float rho0 = const_props[0].rho0;
        float uref = const_props[0].uref;

        const float *f = const_props[0].f_grid + grid_size * 9 * j + 9 * i;

        float u[2]{0.f, 0.f};

        for (int alpha = 0; alpha < 9; alpha++) {
            float f_alpha = f[alpha];
            u[0] += (float) c[alpha][0] * f_alpha;
            u[1] += (float) c[alpha][1] * f_alpha;
        }

        rho /= rho0;
        u[0] = uref / lat_uref * (u[0] / rho);
        u[1] = uref / lat_uref * (u[1] / rho);
        rho *= rho0;

        float force_x = *(const_props[0].force_grid  + grid_size * 2 * j + 2 * i + 0);
        u[0] += force_x * const_props[0].dt / (2 * rho);
        float force_y = *(const_props[0].force_grid  + grid_size * 2 * j + 2 * i + 1);
        u[1] += force_y * const_props[0].dt / (2 * rho);

        *(const_props[0].u_grid + grid_size * 2 * j + 2 * i + 0) = u[0];
        *(const_props[0].u_grid + grid_size * 2 * j + 2 * i + 1) = u[1];
#endif

    }

    __global__
    void boundaryDampKernel() {


        int grid_size = const_props[0].grid_size;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        if (j >= grid_size)
            return;

#ifdef __CUDA_ARCH__
        for (int i = 0; i < grid_size; i += (grid_size - 1)) {
            int step;
            if (i == 0)
                step = 1;
            else
                step = -1;

            i += step;

            const_props[0].u_grid[grid_size * 2 * j + 2 * (i - step) + 0] = const_props[0].u_grid[grid_size * 2 * j +
                                                                                                  2 * i + 0];
            const_props[0].u_grid[grid_size * 2 * j + 2 * (i - step) + 1] = const_props[0].u_grid[grid_size * 2 * j +
                                                                                                  2 * i + 1];
            const_props[0].rho_grid[grid_size * j + (i - step)] = const_props[0].rho_grid[grid_size * j + i];

            const_props[0].u_grid[grid_size * 2 * (i - step) + 2 * j + 0] = const_props[0].u_grid[grid_size * 2 * i +
                                                                                                  2 * j + 0];
            const_props[0].u_grid[grid_size * 2 * (i - step) + 2 * j + 1] = const_props[0].u_grid[grid_size * 2 * i +
                                                                                                  2 * j + 1];
            const_props[0].rho_grid[grid_size * (i - step) + j] = const_props[0].rho_grid[grid_size * i + j];

            i -= step;

            for (int alpha = 0; alpha < 9; alpha++) {
                const_props[0].f_grid[grid_size * 9 * j + 9 * i + alpha] = calcEquilibrium(alpha, i, j);
                const_props[0].f_grid[grid_size * 9 * i + 9 * j + alpha] = calcEquilibrium(alpha, j, i);
            }
        }
#endif
    }

} // namespace jfs