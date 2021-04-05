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
        cudaLBMSolver solver = *( (cudaLBMSolver*) *gpu_this_ptr );

        __syncthreads();

        int N = solver.N;

        if (j >= N || k >= N)
            return;
            
        for (int i=0; i < 9; i++)
            solver.f[N*9*k + 9*j + i] = solver.calc_fbari(i, j, k);
    #endif
}

__global__
void collideKernel(void** gpu_this_ptr, bool* flag_ptr)
{
    int indices[2]{
        (int) (blockIdx.x * blockDim.x + threadIdx.x),
        (int) (blockIdx.y * blockDim.y + threadIdx.y)
    };
    int& j = indices[0];
    int& k = indices[1];

    #ifdef __CUDA_ARCH__
        cudaLBMSolver solver = *( (cudaLBMSolver*) *gpu_this_ptr );

        // __shared__ float f[256*9];
        // float * f_tmp;
        // __shared__ float rho[256];
        // float * rho_tmp;
        // __shared__ float U[256*2];
        // float * U_tmp;
        // __shared__ float F[256*2];
        // float * F_tmp;

        // collide
        int N = solver.N;
        float tau = solver.tau;

        if (j >= N || k >= N)
            return;

        float data[9];
        
        // solver.indexGrid(data, indices, solver.f, SCALAR_FIELD, 9);
        // j -= blockIdx.x * blockDim.x;
        // k -= blockIdx.y * blockDim.y;
        // solver.N = 16;
        // solver.insertIntoGrid(indices, data, f, SCALAR_FIELD, 9);
        // f_tmp = solver.f;
        // solver.f = f;
        // j += blockIdx.x * blockDim.x;
        // k += blockIdx.y * blockDim.y;
        // solver.N = N;
        
        // solver.indexGrid(data, indices, solver.rho_, SCALAR_FIELD, 1);
        // j -= blockIdx.x * blockDim.x;
        // k -= blockIdx.y * blockDim.y;
        // solver.N = 16;
        // solver.insertIntoGrid(indices, data, rho, SCALAR_FIELD, 1);
        // rho_tmp = solver.f;
        // solver.rho_ = rho;
        // j += blockIdx.x * blockDim.x;
        // k += blockIdx.y * blockDim.y;
        // solver.N = N;
        
        // solver.indexGrid(data, indices, solver.U, VECTOR_FIELD, 2);
        // j -= blockIdx.x * blockDim.x;
        // k -= blockIdx.y * blockDim.y;
        // solver.N = 16;
        // solver.insertIntoGrid(indices, data, U, VECTOR_FIELD, 2);
        // U_tmp = solver.U;
        // solver.U = U;
        // j += blockIdx.x * blockDim.x;
        // k += blockIdx.y * blockDim.y;
        // solver.N = N;
        
        // solver.indexGrid(data, indices, solver.F, VECTOR_FIELD, 2);
        // j -= blockIdx.x * blockDim.x;
        // k -= blockIdx.y * blockDim.y;
        // solver.N = 16;
        // solver.insertIntoGrid(indices, data, F, VECTOR_FIELD, 2);
        // F_tmp = solver.F;
        // solver.F = F;

        float* f_jk = data;
        solver.indexGrid(f_jk, indices, solver.f, SCALAR_FIELD, 9);

        for (int i=0; i<9; i++)
        {
            float fi;
            float fbari;
            float Omegai;
            float Fi;
            
            fi = f_jk[i];

            fbari = solver.calc_fbari(i, j, k);            
                
            Fi = solver.calc_Fi(i, j, k);

            Omegai = -(fi - fbari)/tau;

            f_jk[i] = fi + Omegai + Fi;
        }
        solver.insertIntoGrid(indices, f_jk, solver.f, SCALAR_FIELD, 9);
        
        // solver.indexGrid(data, indices, f, SCALAR_FIELD, 9);
        // j += blockIdx.x * blockDim.x;
        // k += blockIdx.y * blockDim.y;
        // solver.N = N;
        // solver.insertIntoGrid(indices, data, f_tmp, SCALAR_FIELD, 9);
        // j -= blockIdx.x * blockDim.x;
        // k -= blockIdx.y * blockDim.y;
        // solver.N = 16;
        
        // solver.indexGrid(data, indices, rho, SCALAR_FIELD, 1);
        // j += blockIdx.x * blockDim.x;
        // k += blockIdx.y * blockDim.y;
        // solver.N = N;
        // solver.insertIntoGrid(indices, data, rho_tmp, SCALAR_FIELD, 1);
        // j -= blockIdx.x * blockDim.x;
        // k -= blockIdx.y * blockDim.y;
        // solver.N = 16;
        
        // solver.indexGrid(data, indices, U, VECTOR_FIELD, 2);
        // j += blockIdx.x * blockDim.x;
        // k += blockIdx.y * blockDim.y;
        // solver.N = N;
        // solver.insertIntoGrid(indices, data, U_tmp, VECTOR_FIELD, 2);
        // j -= blockIdx.x * blockDim.x;
        // k -= blockIdx.y * blockDim.y;
        // solver.N = 16;
        
        // solver.indexGrid(data, indices, F, VECTOR_FIELD, 2);
        // j += blockIdx.x * blockDim.x;
        // k += blockIdx.y * blockDim.y;
        // solver.N = N;
        // solver.insertIntoGrid(indices, data, F_tmp, VECTOR_FIELD, 2);
         
    #endif
}

__global__
void streamKernel(void** gpu_this_ptr, bool* flag_ptr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    #ifdef __CUDA_ARCH__
        cudaLBMSolver solver = *( (cudaLBMSolver*) *gpu_this_ptr );

        __syncthreads();

        // stream
        int N = solver.N;
        const float (*c)[2] = solver.c;
        const float* bounce_back_indices = solver.bounce_back_indices_;

        if (j >= N || k >= N)
            return;

        for (int i=0; i<9; i++)
        {
            float fiStar;

            int cix = c[i][0];
            int ciy = c[i][1];

            if ((k-ciy) >= 0 && (k-ciy) < N && (j-cix) >= 0 && (j-cix) < N)
                fiStar = solver.f0[N*9*(k-ciy) + 9*(j-cix) + i];
            else
            {
                int i_bounce = bounce_back_indices[i];
                fiStar = solver.f0[N*9*k + 9*j + i_bounce];
            }

            solver.f[N*9*k + 9*j + i] = fiStar; 

            float u[2];
            int indices[2]{j, k};
            solver.indexGrid(u, indices, solver.U, VECTOR_FIELD);
            if (std::isinf(u[0]) || std::isinf(u[1]) || std::isnan(u[0]) || std::isnan(u[1]))
                *flag_ptr = true;
        }

        solver.calcPhysicalVals(j, k);
    #endif
}

__global__
void boundaryDampKernel(void** gpu_this_ptr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    #ifdef __CUDA_ARCH__
        cudaLBMSolver solver = *( (cudaLBMSolver*) *gpu_this_ptr );

        __syncthreads();

        int N = solver.N;

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
            solver.indexGrid(u, indices, solver.U, VECTOR_FIELD);
            float rho;
            solver.indexGrid(&rho, indices, solver.rho_, SCALAR_FIELD);

            indices[0] = i;

            solver.insertIntoGrid(indices, &rho, solver.rho_, SCALAR_FIELD);
            solver.insertIntoGrid(indices, u, solver.U, VECTOR_FIELD, 1);

            float fbar[9];
            for (int k = 0; k < 9; k++)
            {
                fbar[k] = solver.calc_fbari(k, i, j);
            }

            solver.insertIntoGrid(indices, fbar, solver.f, SCALAR_FIELD, 9);

            solver.calcPhysicalVals(i, j);
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
            solver.indexGrid(u, indices, solver.U, VECTOR_FIELD);
            float rho;
            solver.indexGrid(&rho, indices, solver.rho_, SCALAR_FIELD);

            indices[1] = i;

            solver.insertIntoGrid(indices, &rho, solver.rho_, SCALAR_FIELD);
            solver.insertIntoGrid(indices, u, solver.U, VECTOR_FIELD, 1);

            float fbar[9];
            for (int k = 0; k < 9; k++)
            {
                fbar[k] = solver.calc_fbari(k, j, i);
            }

            solver.insertIntoGrid(indices, fbar, solver.f, SCALAR_FIELD, 9);

            solver.calcPhysicalVals(j, i);
            
        }
    #endif
}

} // namespace jfs