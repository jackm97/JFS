#include "lbm_solver_cuda.h"

#include <jfs/cuda/lbm_cuda_kernels.cu>

namespace jfs {

    JFS_INLINE CudaLBMSolver::CudaLBMSolver(uint grid_size, float grid_length, BoundType btype, float rho0,
                                            float visc, float uref) :
            cs_{1 / sqrtf(3)} {
        Initialize(grid_size, grid_length, btype, rho0, visc, uref);
    }

    JFS_INLINE void
    CudaLBMSolver::Initialize(uint grid_size, float grid_length, BoundType btype, float rho0, float visc,
                              float uref) {
        grid_size_ = grid_size;
        grid_length_ = grid_length;
        btype_ = btype;

        rho0_ = rho0;
        visc_ = visc;
        uref_ = uref;

        // lattice scaling stuff
        us_ = cs_ / lat_uref_ * uref_;

        dx_ = grid_length_ / (float) (grid_size_ - 1);
        float lat_visc = lat_uref_ / (uref_ * dx_) * visc;
        lat_tau_ = (3.f * lat_visc + .5f);
        dt_ = lat_uref_ / uref_ * dx_ * lat_dt_;

        f_grid_.Resize(grid_size_, 9);
        f0_grid_.Resize(grid_size_, 9);

        rho_grid_.Resize(grid_size_, 1);
        mapped_rho_grid_.Resize(grid_size_, 3);

        u_grid_.Resize(grid_size_, 1);

        force_grid_.Resize(grid_size_, 1);

        ResetFluid();
    }

    JFS_INLINE void CudaLBMSolver::ResetFluid() {
        LBMSolverProps props = SolverProps();
        cudaMemcpyToSymbol(device_solver_props, &props, sizeof(LBMSolverProps), 0, cudaMemcpyHostToDevice);
        current_cuda_lbm_solver = this;
        cudaDeviceSynchronize();

        for (int d = 0; d < 2; d++) {
            u_grid_.SetGridToValue(0, 0, d);
            force_grid_.SetGridToValue(0, 0, d);
        }
        rho_grid_.SetGridToValue(rho0_, 0, 0);

        int threads_per_block = 256;
        int num_blocks = (9 * (int) grid_size_ * (int) grid_size_) / threads_per_block + 1;
        resetDistributionKernel <<<num_blocks, threads_per_block>>>(f_grid_.Data());
        cudaDeviceSynchronize();

        f0_grid_ = f_grid_;

        time_ = 0;
    }

    JFS_INLINE bool CudaLBMSolver::CalcNextStep(const std::vector<Force> &forces) {
        bool failed_step = false;

        for (int d = 0; d < 2; d++){
            force_grid_.SetGridToValue(0, 0, d, CudaGridAsync);
        }

        cudaDeviceSynchronize();

        for (const auto &i : forces) {
            float force[2] = {
                    i.force[0],
                    i.force[1],
            };
            float point[2] = {
                    i.pos[0] / dx_,
                    i.pos[1] / dx_,
            };

            if (point[0] < (float) grid_size_ && point[0] >= 0 && point[1] < (float) grid_size_ && point[1] >= 0)
                for (int d = 0; d < 2; d++) {
                    force_grid_.InterpToGrid(force[d], point[0], point[1], 0, d, CudaGridAsync);
                }
        }

        cudaDeviceSynchronize();

        failed_step = CalcNextStep();

        if (failed_step) ResetFluid();

        host_synced_ = false;

        return failed_step;
    }

    JFS_INLINE void CudaLBMSolver::ForceVelocity(int *i, int *j, float *ux, float *uy, int num_points) {
        if (current_cuda_lbm_solver != this) {
            LBMSolverProps props = SolverProps();
            cudaMemcpyToSymbol(device_solver_props, &props, sizeof(LBMSolverProps), 0, cudaMemcpyHostToDevice);
            current_cuda_lbm_solver = this;
        }

        int *device_i, *device_j;
        cudaMalloc(&device_i, num_points * sizeof(int));
        cudaMemcpy(device_i, i, num_points * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&device_j, num_points * sizeof(int));
        cudaMemcpy(device_j, j, num_points * sizeof(int), cudaMemcpyHostToDevice);

        float *device_ux, *device_uy;
        cudaMalloc(&device_ux, num_points * sizeof(float));
        cudaMemcpy(device_ux, ux, num_points * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&device_uy, num_points * sizeof(float));
        cudaMemcpy(device_uy, uy, num_points * sizeof(int), cudaMemcpyHostToDevice);

        int threads_per_block = 16;
        int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;
        forceVelocityKernel <<<num_blocks, threads_per_block>>>(device_i, device_j, device_ux, device_uy, 0);
        cudaDeviceSynchronize();

        cudaFree(device_i);
        cudaFree(device_j);
        cudaFree(device_ux);
        cudaFree(device_uy);
    }

    JFS_INLINE void CudaLBMSolver::AddMassSource(int i, int j, float rho) {
        if (current_cuda_lbm_solver != this) {
            LBMSolverProps props = SolverProps();
            cudaMemcpyToSymbol(device_solver_props, &props, sizeof(LBMSolverProps), 0, cudaMemcpyHostToDevice);
            current_cuda_lbm_solver = this;
        }

        addMassKernel <<<1, 1>>>(i, j, rho);
        cudaDeviceSynchronize();
    }

    JFS_INLINE void CudaLBMSolver::SetDensityMapping(float min_rho, float max_rho) {
        min_rho_map_ = min_rho;
        max_rho_map_ = max_rho;
    }

    JFS_INLINE void CudaLBMSolver::DensityExtrema(float minmax_rho[2]) {
        minmax_rho[0] = 1.e20;
        minmax_rho[1] = -1.e20;

        rho_grid_.SyncHostWithDevice();

        float* host_rho_data = rho_grid_.HostData();

        for (int idx = 0; idx < grid_size_*grid_size_; idx++){
            if (host_rho_data[idx] < minmax_rho[0])
                minmax_rho[0] = host_rho_data[idx];
            if (host_rho_data[idx] > minmax_rho[1])
                minmax_rho[1] = host_rho_data[idx];
        }

//        float *device_minmax_rho;
//        cudaMalloc(&device_minmax_rho, 2 * sizeof(float));
//        cudaMemcpy(device_minmax_rho, minmax_rho, 2 * sizeof(float), cudaMemcpyHostToDevice);
//
//        int threads_per_block = 256;
//        int num_blocks = ((int) grid_size_ * (int) grid_size_ + threads_per_block - 1) / threads_per_block;
//        getMinMaxRho<<<num_blocks, threads_per_block>>>(device_minmax_rho + 0, device_minmax_rho + 1);
//        cudaDeviceSynchronize();
//
//        cudaMemcpy(minmax_rho, device_minmax_rho, 2 * sizeof(float), cudaMemcpyDeviceToHost);
//        cudaFree(device_minmax_rho);
    }

    JFS_INLINE bool CudaLBMSolver::CalcNextStep() {
        LBMSolverProps props{};
        if (current_cuda_lbm_solver != this) {
            props = SolverProps();
            cudaMemcpyToSymbol(device_solver_props, &props, sizeof(LBMSolverProps), 0, cudaMemcpyHostToDevice);
            current_cuda_lbm_solver = this;
        }

        bool failed_step = false;
        bool *flag_ptr;
        cudaMalloc(&flag_ptr, sizeof(bool));
        cudaMemcpy(flag_ptr, &failed_step, sizeof(bool), cudaMemcpyHostToDevice);

        int threads_per_block = 256;
        int num_blocks = (9 * (int) grid_size_ * (int) grid_size_ + threads_per_block - 1) / threads_per_block;

        streamKernel <<<num_blocks, threads_per_block>>>(flag_ptr);
        cudaDeviceSynchronize();

        num_blocks = ((int) grid_size_ * (int) grid_size_ + threads_per_block - 1) / threads_per_block;
        calcRhoKernel <<<num_blocks, threads_per_block>>>();
        cudaDeviceSynchronize();
        calcVelocityKernel <<<num_blocks, threads_per_block>>>();
        cudaDeviceSynchronize();

        // do any field manipulations before next step
        if (btype_ == DAMPED) {
            boundaryDampKernel << <(grid_size_ + threads_per_block - 1) / threads_per_block, threads_per_block >> > ();
            cudaDeviceSynchronize();
        }

        num_blocks = (9 * (int) grid_size_ * (int) grid_size_ + threads_per_block - 1) / threads_per_block;
        collideKernel <<<num_blocks, threads_per_block>>>(flag_ptr);
        cudaDeviceSynchronize();

        time_ += dt_;

        cudaMemcpy(&failed_step, flag_ptr, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaFree(flag_ptr);
        return failed_step;
    }

    __host__
    JFS_INLINE void CudaLBMSolver::MapDensity() {
        float minmax_rho[2];
        DensityExtrema(minmax_rho);

        float min_rho_map, max_rho_map;

        if (min_rho_map_ >= 0)
            min_rho_map = min_rho_map_;
        else
            min_rho_map = minmax_rho[0];

        if (max_rho_map_ >= 0)
            max_rho_map = max_rho_map_;
        else
            max_rho_map = minmax_rho[1];

        int threads_per_block = 256;
        int num_blocks = ((int) grid_size_ * (int) grid_size_ + threads_per_block - 1) / threads_per_block;
        mapDensityKernel<<<num_blocks, threads_per_block>>>(mapped_rho_grid_.Data(), min_rho_map, max_rho_map);
        cudaDeviceSynchronize();
//        float *host_rho_data = RhoData();
//
//        float min_rho = host_rho_data[0];
//        float max_rho = host_rho_data[0];
//        float mean_rho = 0;
//        for (int i = 0; i < grid_size_ * grid_size_; i++)
//            mean_rho += host_rho_data[i];
//        mean_rho /= (float) grid_size_ * (float) grid_size_;
//
//        for (int i = 0; i < grid_size_ * grid_size_ && min_rho_map_ == -1; i++) {
//            if (host_rho_data[i] < min_rho)
//                min_rho = host_rho_data[i];
//        }
//
//        if (min_rho_map_ != -1)
//            min_rho = min_rho_map_;
//
//        for (int i = 0; i < grid_size_ * grid_size_ && max_rho_map_ == -1; i++) {
//            if (host_rho_data[i] > max_rho)
//                max_rho = host_rho_data[i];
//        }
//
//        if (max_rho_map_ == -1 && min_rho_map_ == -1) {
//            if (max_rho - mean_rho > mean_rho - min_rho)
//                min_rho = mean_rho - (max_rho - mean_rho);
//            else
//                max_rho = mean_rho - (min_rho - mean_rho);
//        }
//
//        if (max_rho_map_ != -1)
//            max_rho = max_rho_map_;
//
//        float *host_mapped_rho_data = mapped_rho_grid_.HostData();
//        for (int i = 0; i < grid_size_; i++)
//            for (int j = 0; j < grid_size_; j++) {
//                float rho_mapped;
//                rho_mapped = host_rho_data[grid_size_ * j + i];
//                if ((max_rho - min_rho) != 0)
//                    rho_mapped = (rho_mapped - min_rho) / (max_rho - min_rho);
//                else
//                    rho_mapped = 0 * rho_mapped;
//
//                // rho_mapped = (rho_mapped < 0) ? 0 : rho_mapped;
//                // rho_mapped = (rho_mapped > 1) ? 1 : rho_mapped;
//
//                host_mapped_rho_data[grid_size_ * 3 * j + 3 * i + 0] = rho_mapped;
//                host_mapped_rho_data[grid_size_ * 3 * j + 3 * i + 1] = rho_mapped;
//                host_mapped_rho_data[grid_size_ * 3 * j + 3 * i + 2] = rho_mapped;
//            }
//
////        mapped_rho_grid_.SyncDeviceWithHost();
    }

    JFS_INLINE LBMSolverProps CudaLBMSolver::SolverProps() {

        LBMSolverProps props{};
        props.grid_size = grid_size_;
        props.grid_length = grid_length_;
        props.btype = btype_;
        props.rho0 = rho0_;
        props.lat_tau = lat_tau_;
        props.uref = uref_;
        props.dt = dt_;
        props.rho_grid = rho_grid_.Data();
        props.f_grid = f_grid_.Data();
        props.f0_grid = f0_grid_.Data();
        props.u_grid = u_grid_.Data();
        props.force_grid = force_grid_.Data();

        return props;
    }

} // namespace jfs