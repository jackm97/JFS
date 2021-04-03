#include <jfs/cuda/lbm_cuda_kernels.h>
#include <cuda_runtime.h>

namespace jfs {

__global__
void setupGPULBM(void** gpu_this_ptr, LBMSolverProps props)
{
    #ifdef __CUDA_ARCH__
        *gpu_this_ptr = new cudaLBMSolver();
        cudaLBMSolver* solver_ptr = (cudaLBMSolver*) *gpu_this_ptr;
        solver_ptr->initialize(props.N, props.L, props.btype, props.iter_per_frame, props.rho0, props.visc, props.uref);

        solver_ptr->rho_ = props.rho;
        solver_ptr->rho_mapped_ = props.rho_mapped;
        solver_ptr->f = props.f;
        solver_ptr->f0 = props.f0;
        solver_ptr->U = props.U;
        solver_ptr->F = props.F;
    #endif
}

__global__
void freeGPULBM(void** gpu_this_ptr)
{
    #ifdef __CUDA_ARCH__
        cudaLBMSolver* solver_ptr = (cudaLBMSolver*) *gpu_this_ptr;
        delete solver_ptr;
    #endif
}

__global__
void forceVelocityKernel(int i, int j, float ux, float uy, void** gpu_this_ptr)
{
    #ifdef __CUDA_ARCH__
        cudaLBMSolver* solver_ptr = (cudaLBMSolver*) *gpu_this_ptr;
        solver_ptr->forceVelocity(i, j, ux, uy);
    #endif
}

__global__
void resetDistributionKernel(void** gpu_this_ptr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    #ifdef __CUDA_ARCH__
        cudaLBMSolver* solver_ptr = (cudaLBMSolver*) *gpu_this_ptr;

        int N = solver_ptr->N;

        if (j >= N || k >= N)
            return;
            
        for (int i=0; i < 9; i++)
            solver_ptr->f[N*9*k + 9*j + i] = solver_ptr->calc_fbari(i, j, k);
    #endif
}

__global__
void collideKernel(void** gpu_this_ptr, bool* flag_ptr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    #ifdef __CUDA_ARCH__
        cudaLBMSolver* solver_ptr = (cudaLBMSolver*) *gpu_this_ptr;

        // collide
        int N = solver_ptr->N;
        float tau = solver_ptr->tau;

        if (j >= N || k >= N)
            return;

        for (int i=0; i<9; i++)
        {
            float fi;
            float fbari;
            float Omegai;
            float Fi;
            
            fi = solver_ptr->f[N*9*k + 9*j + i];

            fbari = solver_ptr->calc_fbari(i, j, k);            
                
            Fi = solver_ptr->calc_Fi(i, j, k);

            Omegai = -(fi - fbari)/tau;

            solver_ptr->f[N*9*k + 9*j + i] = fi + Omegai + Fi;
        }
    #endif
}

__global__
void streamKernel(void** gpu_this_ptr, bool* flag_ptr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    #ifdef __CUDA_ARCH__
        cudaLBMSolver* solver_ptr = (cudaLBMSolver*) *gpu_this_ptr;

        // stream
        int N = solver_ptr->N;
        const float (*c)[2] = solver_ptr->c;
        const float* bounce_back_indices = solver_ptr->bounce_back_indices_;

        if (j >= N || k >= N)
            return;

        for (int i=0; i<9; i++)
        {
            float fiStar;

            int cix = c[i][0];
            int ciy = c[i][1];

            if ((k-ciy) >= 0 && (k-ciy) < N && (j-cix) >= 0 && (j-cix) < N)
                fiStar = solver_ptr->f0[N*9*(k-ciy) + 9*(j-cix) + i];
            else
            {
                int i_bounce = bounce_back_indices[i];
                fiStar = solver_ptr->f0[N*9*k + 9*j + i_bounce];
            }

            solver_ptr->f[N*9*k + 9*j + i] = fiStar; 

            float u[2];
            int indices[2]{j, k};
            solver_ptr->indexGrid(u, indices, solver_ptr->U, VECTOR_FIELD);
            if (std::isinf(u[0]) || std::isinf(u[1]) || std::isnan(u[0]) || std::isnan(u[1]))
                *flag_ptr = true;
        }

        solver_ptr->calcPhysicalVals(j, k);
    #endif
}

__global__
void boundaryDampKernel(void** gpu_this_ptr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    #ifdef __CUDA_ARCH__
        cudaLBMSolver* solver_ptr = (cudaLBMSolver*) *gpu_this_ptr;

        int N = solver_ptr->N;

        if (j >= N)
            return;

        for (int i = 0; i < N; i+=(N-1))
        {
            int step;
            if (i == 0)
                step = 1;
            else
                step = -1;
            
            int indices[2]{
                i + step,
                j
            };
            float u[2];
            solver_ptr->indexGrid(u, indices, solver_ptr->U, VECTOR_FIELD);
            float rho;
            solver_ptr->indexGrid(&rho, indices, solver_ptr->rho_, SCALAR_FIELD);

            indices[0] = i;

            solver_ptr->insertIntoGrid(indices, &rho, solver_ptr->rho_, SCALAR_FIELD);
            solver_ptr->insertIntoGrid(indices, u, solver_ptr->U, VECTOR_FIELD, 1);

            float fbar[9];
            for (int k = 0; k < 9; k++)
            {
                fbar[k] = solver_ptr->calc_fbari(k, i, j);
            }

            solver_ptr->insertIntoGrid(indices, fbar, solver_ptr->f, SCALAR_FIELD, 9);

            solver_ptr->calcPhysicalVals(i, j);
        } 

        for (int i = 0; i < N; i+=(N-1))
        {
            int step;
            if (i == 0)
                step = 1;
            else
                step = -1;

            int indices[2]{
                j,
                i + step
            };
            float u[2];
            solver_ptr->indexGrid(u, indices, solver_ptr->U, VECTOR_FIELD);
            float rho;
            solver_ptr->indexGrid(&rho, indices, solver_ptr->rho_, SCALAR_FIELD);

            indices[1] = i;

            solver_ptr->insertIntoGrid(indices, &rho, solver_ptr->rho_, SCALAR_FIELD);
            solver_ptr->insertIntoGrid(indices, u, solver_ptr->U, VECTOR_FIELD, 1);

            float fbar[9];
            for (int k = 0; k < 9; k++)
            {
                fbar[k] = solver_ptr->calc_fbari(k, j, i);
            }

            solver_ptr->insertIntoGrid(indices, fbar, solver_ptr->f, SCALAR_FIELD, 9);

            solver_ptr->calcPhysicalVals(j, i);
            
        }
    #endif
}

} // namespace jfs