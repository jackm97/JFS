#include <jfs/cuda/lbm_solver_cuda.h>
#include <jfs/cuda/lbm_cuda_kernels.h>
#include <cuda_runtime.h>

#include <cstring>
#include <iostream>

namespace jfs {

__host__
JFS_INLINE cudaLBMSolver::cudaLBMSolver(unsigned int N, float L, BoundType btype, int iter_per_frame, float rho0, float visc, float uref)
{
    initialize(N, L, btype, iter_per_frame, rho0, visc, us);
}

__host__ __device__
JFS_INLINE void cudaLBMSolver::initialize(unsigned int N, float L, BoundType btype, int iter_per_frame, float rho0, float visc, float uref)
{
    this->iter_per_frame = iter_per_frame;
    this->rho0 = rho0;
    this->visc = visc;
    this->uref = uref;
    
    // lattice scaling stuff
    this->cs = 1/sqrtf(3);
    this->us = cs/urefL * uref;

    // dummy dt because it is calculated
    float dummy_dt = 0;

    initializeGrid(N, L, btype, dummy_dt);
    
    this->viscL = urefL/(uref * dx) * visc;
    this->tau = (3*viscL + .5);
    this->dt = urefL/uref * dx * dtL;

    #ifndef __CUDA_ARCH__

        clearGrid();

        cudaMalloc(&f, 9*N*N*sizeof(float));
        cudaMalloc(&f0, 9*N*N*sizeof(float));

        cudaMalloc(&rho_, N*N*sizeof(float));
        cpu_rho_ = new float[N*N];
        cudaMalloc(&rho_mapped_, 3*N*N*sizeof(float));
        cpu_rho_mapped_ = new float[3*N*N];

        cudaMalloc(&U, 2*N*N*sizeof(float));
        cpu_U_ = new float[2*N*N];

        cudaMalloc(&F, 2*N*N*sizeof(float));

        LBMSolverProps props;
        props.N = N;
        props.L = L;
        props.btype = btype;
        props.iter_per_frame = iter_per_frame;
        props.rho0 = rho0;
        props.visc = visc;
        props.uref = uref;
        props.rho = rho_;
        props.rho_mapped = rho_mapped_;
        props.f = f;
        props.f0 = f0;
        props.U = U;
        props.F = F;
        
        cudaMalloc(&gpu_this_ptr, sizeof(void*));
        setupGPULBM<<<1, 1>>>(gpu_this_ptr, props);
        cudaDeviceSynchronize();

        if (N!=1)
            resetFluid();

        is_initialized_ = true;

    #endif
}

__host__
JFS_INLINE void cudaLBMSolver::resetFluid()
{
    float* field_tmp = new float[2*N*N];
    for (int i = 0; i < 2*N*N; i++)
        field_tmp[i] = 0;
    cudaMemcpy(U, field_tmp, 2*N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F, field_tmp, 2*N*N*sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < N*N; i++)
        field_tmp[i] = rho0;
    cudaMemcpy(rho_, field_tmp, N*N*sizeof(float), cudaMemcpyHostToDevice);
    delete [] field_tmp;

    dim3 threads_per_block(16, 16);
    dim3 num_blocks(N / threads_per_block.x + 1, N / threads_per_block.y + 1);
    resetDistributionKernel<<<num_blocks, threads_per_block>>>(gpu_this_ptr);
    cudaDeviceSynchronize();

    T = 0;
}

__host__
JFS_INLINE bool cudaLBMSolver::calcNextStep(const std::vector<Force> forces)
{
    bool failedStep = false;
    
    float* field_tmp = new float[2*N*N];
    try
    {   
        cudaMemcpy(field_tmp, F, 2*N*N*sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < forces.size(); i++)
        {
            float force[3] = {
                forces[i].force[0],
                forces[i].force[1],
                forces[i].force[2]
            };
            float point[3] = {
                forces[i].pos[0]/this->D,
                forces[i].pos[1]/this->D,
                forces[i].pos[2]/this->D
            };
            this->interpPointToGrid(force, point, field_tmp, VECTOR_FIELD, 1, Add);
        }
        cudaMemcpy(F, field_tmp, 2*N*N*sizeof(float), cudaMemcpyHostToDevice);
        
        for (int iter = 0; iter < iter_per_frame; iter++)
        {
            failedStep = calcNextStep();
            if (failedStep)
                break;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        failedStep = true;
    }
    for (int i = 0; i < 2*N*N; i++)
        field_tmp[i] = 0;
    cudaMemcpy(F, field_tmp, 2*N*N*sizeof(float), cudaMemcpyHostToDevice);
    delete [] field_tmp;

    if (failedStep) resetFluid();

    return failedStep;
}

__host__ __device__
JFS_INLINE void cudaLBMSolver::forceVelocity(int i, int j, float ux, float uy)
{
    #ifndef __CUDA_ARCH__
        forceVelocityKernel<<<1, 1>>>(i, j, ux, uy, gpu_this_ptr);
        cudaDeviceSynchronize();
    #else
        int indices[2]{i, j};

        float u[2]{ux, uy};

        float u_prev[2];
        indexGrid(u_prev, indices, U, VECTOR_FIELD);

        float rho;
        indexGrid(&rho, indices, rho_, SCALAR_FIELD);

        float force[2]{
            (u[0] - u_prev[0]) * rho / this->dt,
            (u[1] - u_prev[1]) * rho / this->dt
        };
        insertIntoGrid(indices, force, F, VECTOR_FIELD, 1, Replace);

    #endif
}

__host__
JFS_INLINE void cudaLBMSolver::setDensityMapping(float minrho, float maxrho)
{
    minrho_ = minrho;
    maxrho_ = maxrho;
}

__host__
JFS_INLINE void cudaLBMSolver::densityExtrema(float minmax_rho[2])
{
    rhoData();

    float minrho_ = cpu_rho_[0];
    float maxrho_ = cpu_rho_[0];

    for (int i=0; i < N*N; i++)
    {
        if (cpu_rho_[i] < minrho_)
            minrho_ = cpu_rho_[i];
    }

    for (int i=0; i < N*N; i++)
    {
        if (cpu_rho_[i] > maxrho_)
            maxrho_ = cpu_rho_[i];
    }

    minmax_rho[0] = minrho_;
    minmax_rho[1] = maxrho_;
}

__host__
JFS_INLINE float* cudaLBMSolver::rhoData()
{
    cudaMemcpy(cpu_rho_, rho_, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    return cpu_rho_;
}

__host__
JFS_INLINE float* cudaLBMSolver::mappedRhoData()
{
    mapDensity();
    
    cudaMemcpy(cpu_rho_mapped_, rho_mapped_, 3*N*N*sizeof(float), cudaMemcpyDeviceToHost);

    return cpu_rho_mapped_;
}

__host__
JFS_INLINE float* cudaLBMSolver::velocityData()
{
    cudaMemcpy(cpu_U_, U, 2*N*N*sizeof(float), cudaMemcpyDeviceToHost);

    return cpu_U_;
}

__host__
JFS_INLINE bool cudaLBMSolver::calcNextStep()
{
    bool failed_step = false;

    bool* flag_ptr;
    cudaMalloc(&flag_ptr, sizeof(bool));
    cudaMemcpy(flag_ptr, &failed_step, sizeof(bool), cudaMemcpyHostToDevice);

    dim3 threads_per_block(16, 16);
    dim3 num_blocks(N / threads_per_block.x + 1, N / threads_per_block.y + 1);
    collideKernel<<<num_blocks, threads_per_block>>>(gpu_this_ptr, flag_ptr);
    cudaDeviceSynchronize();

    cudaMemcpy(f0, f, 9*N*N*sizeof(float), cudaMemcpyDeviceToDevice);

    streamKernel<<<num_blocks, threads_per_block>>>(gpu_this_ptr, flag_ptr);
    cudaDeviceSynchronize();

    cudaMemcpy(&failed_step, flag_ptr, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(flag_ptr);

    // do any field manipulations before collision step
    if (bound_type_ == DAMPED)
        doBoundaryDamping();

    T += dt;
    
    return failed_step;
}

__device__
JFS_INLINE float cudaLBMSolver::calc_Fi(int i, int j, int k)
{
    int indices[2]{j, k};
    
    float wi = w[i];

    float u[2];
    indexGrid(u, indices, U, VECTOR_FIELD);
    u[0] *= urefL/uref;
    u[1] *= urefL/uref;
    float ci[2]{c[i][0], c[i][1]};

    float ci_dot_u = ci[0]*u[0] + ci[1]*u[1];

    float F_jk[2];
    indexGrid(F_jk, indices, F, VECTOR_FIELD);
    F_jk[0] *= ( 1/rho0 * dx * powf(urefL/uref,2) );
    F_jk[1] *= ( 1/rho0 * dx * powf(urefL/uref,2) );

    float Fi = (1 - tau/2) * wi * (
         ( (1/powf(cs,2))*(ci[0] - u[0]) + (ci_dot_u/powf(cs,4)) * ci[0] )  * F_jk[0] + 
         ( (1/powf(cs,2))*(ci[1] - u[1]) + (ci_dot_u/powf(cs,4)) * ci[1] )  * F_jk[1]
    );

    return Fi;
}

__device__
JFS_INLINE float cudaLBMSolver::calc_fbari(int i, int j, int k)
{
    int indices[2]{j, k};
    
    float fbari;
    float rho_jk; // rho_ at point P -> (j,k)
    indexGrid(&rho_jk, indices, rho_, SCALAR_FIELD);
    float wi = w[i];

    float u[2];
    indexGrid(u, indices, U, VECTOR_FIELD);
    u[0] *= urefL/uref;
    u[1] *= urefL/uref;
    float ci[2]{c[i][0], c[i][1]};

    float ci_dot_u = ci[0]*u[0] + ci[1]*u[1];
    float u_dot_u = u[0]*u[0] + u[1]*u[1];

    fbari = wi * rho_jk/rho0 * ( 1 + ci_dot_u/(powf(cs,2)) + powf(ci_dot_u,2)/(2*powf(cs,4)) - u_dot_u/(2*powf(cs,2)) );

    return fbari;
}

__host__
JFS_INLINE void cudaLBMSolver::mapDensity()
{
    rhoData();

    float minrho = cpu_rho_[0];
    float maxrho = cpu_rho_[0];
    float meanrho = 0;
    for (int i = 0; i < N*N; i++)
        meanrho += cpu_rho_[i];
    meanrho /= N*N;

    for (int i=0; i < N*N && minrho_ == -1; i++)
    {
        if (cpu_rho_[i] < minrho)
            minrho = cpu_rho_[i];
    }

    if (minrho_ != -1)
        minrho = minrho_;

    for (int i=0; i < N*N && maxrho_ == -1; i++)
    {
        if (cpu_rho_[i] > maxrho)
            maxrho = cpu_rho_[i];
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

    for (int i=0; i < N; i++)
        for (int j=0; j < N; j++)
        {
            int indices[2]{i, j};
            float rho;
            indexGrid(&rho, indices, cpu_rho_, SCALAR_FIELD, 1);
            if ((maxrho - minrho) != 0)
                rho = (rho- minrho)/(maxrho - minrho);
            else
                rho = 0 * rho;

            // rho = (rho < 0) ? 0 : rho;
            // rho = (rho > 1) ? 1 : rho;

            float rho_gray[3]{rho, rho, rho};

            insertIntoGrid(indices, rho_gray, cpu_rho_mapped_, SCALAR_FIELD, 3);
        }

    cudaMemcpy(rho_mapped_, cpu_rho_mapped_, 3*N*N*sizeof(float), cudaMemcpyHostToDevice);
}

__device__
JFS_INLINE void cudaLBMSolver::calcPhysicalVals(int j, int k)
{
    float rho_jk = 0;
    float momentum_jk[2]{0, 0};

    for (int i=0; i<9; i++)
    {
        rho_jk += f[N*9*k + 9*j + i];
        momentum_jk[0] += c[i][0] * f[N*9*k + 9*j + i];
        momentum_jk[1] += c[i][1] * f[N*9*k + 9*j + i];
    }

    float* u = momentum_jk;
    u[0] = uref/urefL * (momentum_jk[0]/rho_jk);
    u[1] = uref/urefL * (momentum_jk[1]/rho_jk);
    rho_jk = rho0 * rho_jk;

    int indices[2]{j, k};

    insertIntoGrid(indices, &rho_jk, rho_, SCALAR_FIELD);
    insertIntoGrid(indices, u, U, VECTOR_FIELD);
}

__host__
JFS_INLINE void cudaLBMSolver::doBoundaryDamping()
{
    int threads_per_block = 256;
    int num_blocks = N / threads_per_block + 1;
    boundaryDampKernel<<<num_blocks, threads_per_block>>>(gpu_this_ptr);
    cudaDeviceSynchronize();
}

__host__ __device__
JFS_INLINE void cudaLBMSolver::clearGrid()
{

    if (!is_initialized_)
        return;
    
    cudaFree(f);
    cudaFree(f0);
    cudaFree(rho_);
    delete [] cpu_rho_;
    cudaFree(rho_mapped_);
    delete [] cpu_rho_mapped_;
    cudaFree(U);
    delete [] cpu_U_;
    cudaFree(F);

    freeGPULBM<<<1, 1>>>(gpu_this_ptr);
    cudaFree(gpu_this_ptr);

    is_initialized_ = false;

    cudaDeviceSynchronize();
}

__host__ __device__
JFS_INLINE cudaLBMSolver::~cudaLBMSolver()
{
    #ifndef __CUDA_ARCH__
        clearGrid();
    #endif
}

__device__
JFS_INLINE void cudaLBMSolver::operator=(const cudaLBMSolver& src)
{
    #ifndef __CUDA_ARCH__
        this->initialize(src.N, src.L, src.bound_type_, src.iter_per_frame, src.rho0, src.visc, src.uref);

        this->rho_ = src.rho_;
        this->rho_mapped_ = src.rho_mapped_;
        this->f = src.f;
        this->f0 = src.f0;
        this->U = src.U;
        this->F = src.F;
    #endif
}

} // namespace jfs