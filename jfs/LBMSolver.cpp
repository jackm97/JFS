#include <jfs/LBMSolver.h>

#include <cmath>
#include <cstring>
#include <iostream>

namespace jfs {

JFS_INLINE LBMSolver::LBMSolver(unsigned int N, float L, BoundType btype, int iter_per_frame, float rho0, float visc, float uref)
{
    initialize(N, L, btype, iter_per_frame, rho0, visc, uref);
}

JFS_INLINE void LBMSolver::initialize(unsigned int N, float L, BoundType btype, int iter_per_frame, float rho0, float visc, float uref)
{
    this->iter_per_frame = iter_per_frame;
    this->rho0 = rho0;
    this->visc = visc;
    this->uref = uref;

    clearGrid();
    
    // lattice scaling stuff
    cs = 1/std::sqrt(3);
    us = cs/urefL * uref;

    // dummy dt because it is calculated
    float dummy_dt;

    initializeGrid(N, L, btype, dummy_dt);
    
    viscL = urefL/(uref * dx) * visc;
    tau = (3*viscL + .5);
    dt = urefL/uref * dx * dtL;

    f = new float[9*N*N];
    f0 = new float[9*N*N];

    rho_ = new float[N*N];
    rho_mapped_ = new float[3*N*N];

    U = new float[2*N*N];

    F = new float[2*N*N];

    resetFluid();

    is_initialized_ = true;
}

JFS_INLINE void LBMSolver::setDensityMapping(float minrho, float maxrho)
{
    minrho_ = minrho;
    maxrho_ = maxrho;
}

JFS_INLINE void LBMSolver::densityExtrema(float minmax_rho[2])
{

    float minrho_ = rho_[0];
    float maxrho_ = rho_[0];

    for (int i=0; i < N*N; i++)
    {
        if (rho_[i] < minrho_)
            minrho_ = rho_[i];
    }

    for (int i=0; i < N*N; i++)
    {
        if (rho_[i] > maxrho_)
            maxrho_ = rho_[i];
    }

    minmax_rho[0] = minrho_;
    minmax_rho[1] = maxrho_;
}

JFS_INLINE void LBMSolver::resetFluid()
{

    for (int i = 0; i < N*N; i++)
    {
        U[2*i + 0] = 0;
        U[2*i + 1] = 0;
        
        F[2*i + 0] = 0;
        F[2*i + 1] = 0;

        rho_[i] = rho0;
    }

    // reset distribution
    for (int j=0; j < N; j++)
        for (int k=0; k < N; k++)
            for (int i=0; i < 9; i++)
                f[N*9*k + 9*j + i] = calc_fbari(i, j, k);

    T = 0;
}

JFS_INLINE void LBMSolver::mapDensity()
{
    auto btype = this->bound_type_;
    auto L = this->L;
    auto N = this->N;
    auto D = this->D;

    float minrho = rho_[0];
    float maxrho = rho_[0];
    float meanrho = 0;
    for (int i = 0; i < N*N; i++)
        meanrho += rho_[i];
    meanrho /= N*N;

    for (int i=0; i < N*N && minrho_ == -1; i++)
    {
        if (rho_[i] < minrho)
            minrho = rho_[i];
    }

    if (minrho_ != -1)
        minrho = minrho_;

    for (int i=0; i < N*N && maxrho_ == -1; i++)
    {
        if (rho_[i] > maxrho)
            maxrho = rho_[i];
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
            indexGrid(&rho, indices, rho_, SCALAR_FIELD, 1);
            if ((maxrho - minrho) != 0)
                rho = (rho- minrho)/(maxrho - minrho);
            else
                rho = 0 * rho;

            // rho = (rho < 0) ? 0 : rho;
            // rho = (rho > 1) ? 1 : rho;

            float rho_gray[3]{rho, rho, rho};

            insertIntoGrid(indices, rho_gray, rho_mapped_, SCALAR_FIELD, 3);
        }
}

JFS_INLINE void LBMSolver::forceVelocity(int i, int j, float ux, float uy)
{

    int indices[2]{i, j};

    float u[2]{ux, uy};

    float u_prev[2];
    indexGrid(u_prev, indices, U, VECTOR_FIELD);

    float rho;
    indexGrid(&rho, indices, rho_, SCALAR_FIELD);

    float f[2]{
        (u[0] - u_prev[0]) * rho / this->dt,
        (u[1] - u_prev[1]) * rho / this->dt
    };
    insertIntoGrid(indices, f, F, VECTOR_FIELD, 1, Add);
}

JFS_INLINE void LBMSolver::doBoundaryDamping()
{
    for (int i = 0; i < N; i+=(N-1))
    {
        int step;
        if (i == 0)
            step = 1;
        else
            step = -1;
        for (int j = 0; j < N; j++)
        {
            int indices[2]{
                i + step,
                j
            };
            float u[2];
            indexGrid(u, indices, U, VECTOR_FIELD);
            float rho;
            indexGrid(&rho, indices, rho_, SCALAR_FIELD);

            indices[0] = i;

            insertIntoGrid(indices, &rho, rho_, SCALAR_FIELD);
            insertIntoGrid(indices, u, U, VECTOR_FIELD, 1);

            float fbar[9];
            for (int k = 0; k < 9; k++)
            {
                fbar[k] = calc_fbari(k, i, j);
            }

            insertIntoGrid(indices, fbar, f, SCALAR_FIELD, 9);

            calcPhysicalVals(i, j);
        }
    } 

    for (int i = 0; i < N; i+=(N-1))
    {
        int step;
        if (i == 0)
            step = 1;
        else
            step = -1;
        for (int j = 0; j < N; j++)
        {
            int indices[2]{
                j,
                i + step
            };
            float u[2];
            indexGrid(u, indices, U, VECTOR_FIELD);
            float rho;
            indexGrid(&rho, indices, rho_, SCALAR_FIELD);

            indices[1] = i;

            insertIntoGrid(indices, &rho, rho_, SCALAR_FIELD);
            insertIntoGrid(indices, u, U, VECTOR_FIELD, 1);

            float fbar[9];
            for (int k = 0; k < 9; k++)
            {
                fbar[k] = calc_fbari(k, j, i);
            }

            insertIntoGrid(indices, fbar, f, SCALAR_FIELD, 9);

            calcPhysicalVals(j, i);
        }
    }
}

JFS_INLINE bool LBMSolver::calcNextStep()
{

    BoundType btype = this->bound_type_;

    int iterations = iter_per_frame;
    for (int iter = 0; iter < iterations; iter++)
    {        
        // collide
        for (int idx=0; idx<(N*N*9); idx++)
        {

            float fi;
            float fbari;
            float Omegai;
            float Fi;

            int idx_tmp = idx;
            int k = idx_tmp / (N*9);
            idx_tmp -= k * (N*9);
            int j = idx_tmp / 9;
            idx_tmp -= j * 9;
            int i = idx_tmp;
            
            fi = f[N*9*k + 9*j + i];

            fbari = calc_fbari(i, j, k);            
                
            Fi = calc_Fi(i, j, k);

            Omegai = -(fi - fbari)/tau;

            f[N*9*k + 9*j + i] = fi + Omegai + Fi;
        }

        std::memcpy(f0, f, 9*N*N*sizeof(float));
        
        // stream
        for (int idx=0; idx<(N*N*9); idx++)
        {
            float fiStar;
            
            int idx_tmp = idx;
            int k = idx_tmp / (N*9);
            idx_tmp -= k * (N*9);
            int j = idx_tmp / 9;
            idx_tmp -= j * 9;
            int i = idx_tmp;

            int cix = c[i][0];
            int ciy = c[i][1];

            if ((k-ciy) >= 0 && (k-ciy) < N && (j-cix) >= 0 && (j-cix) < N)
                fiStar = f0[N*9*(k-ciy) + 9*(j-cix) + i];
            else
            {
                int i_bounce = bounce_back_indices_[i];
                fiStar = f0[N*9*k + 9*j + i_bounce];
            }

            f[N*9*k + 9*j + i] = fiStar; 

            calcPhysicalVals(j, k);

            float u[2];
            int indices[2]{j, k};
            indexGrid(u, indices, U, VECTOR_FIELD);
            if (std::isinf(u[0]) || std::isinf(u[1]) || std::isnan(u[0]) || std::isnan(u[1]))
                return true;
        }

        // do any field manipulations before collision step
        if (btype == DAMPED)
            doBoundaryDamping();

        T += dt;
    }

    return false;
}

JFS_INLINE bool LBMSolver::calcNextStep(const std::vector<Force> forces)
{

    bool failedStep = false;
    try
    {    
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
            this->interpPointToGrid(force, point, F, VECTOR_FIELD, 1, Add);
        }

        failedStep = calcNextStep();

        for (int i = 0; i < N*N; i++)
        {
            F[2*i + 0] = 0;
            F[2*i + 1] = 0;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        failedStep = true;
    }

    if (failedStep) resetFluid();

    return failedStep;
}

JFS_INLINE void LBMSolver::clearGrid()
{

    if (!is_initialized_)
        return;
    
    delete [] f;
    delete [] f0;
    delete [] rho_;
    delete [] rho_mapped_;
    delete [] U;
    delete [] F;

    is_initialized_ = false;
}

JFS_INLINE float LBMSolver::calc_fbari(int i, int j, int k)
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

    fbari = wi * rho_jk/rho0 * ( 1 + ci_dot_u/(std::pow(cs,2)) + std::pow(ci_dot_u,2)/(2*std::pow(cs,4)) - u_dot_u/(2*std::pow(cs,2)) );

    return fbari;
}

JFS_INLINE float LBMSolver::calc_Fi(int i, int j, int k)
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
    F_jk[0] *= ( 1/rho0 * dx * std::pow(urefL/uref,2) );
    F_jk[1] *= ( 1/rho0 * dx * std::pow(urefL/uref,2) );

    float Fi = (1 - tau/2) * wi * (
         ( (1/std::pow(cs,2))*(ci[0] - u[0]) + (ci_dot_u/std::pow(cs,4)) * ci[0] )  * F_jk[0] + 
         ( (1/std::pow(cs,2))*(ci[1] - u[1]) + (ci_dot_u/std::pow(cs,4)) * ci[1] )  * F_jk[1]
    );

    return Fi;
}

JFS_INLINE void LBMSolver::calcPhysicalVals()
{
    float rho_jk = 0;
    float momentum_jk[2]{0, 0};

    for (int j=0; j<N; j++)
    {
        for (int k=0; k<N; k++)
        {
            rho_jk = 0;
            momentum_jk[0] += 0;
            momentum_jk[1] += 0;

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
    }
}

JFS_INLINE void LBMSolver::calcPhysicalVals(int j, int k)
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

} // namespace jfs