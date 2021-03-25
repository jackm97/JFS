#include <jfs/LBMSolver.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace jfs {

JFS_INLINE LBMSolver::LBMSolver(unsigned int N, float L, BoundType btype, int iter_per_frame, float rho0, float visc, float uref)
{
    initialize(N, L, btype, iter_per_frame, rho0, visc, us);
}

JFS_INLINE void LBMSolver::initialize(unsigned int N, float L, BoundType btype, int iter_per_frame, float rho0, float visc, float uref)
{
    this->iter_per_frame = iter_per_frame;
    this->rho0 = rho0;
    this->visc = visc;
    this->uref = uref;

    if (is_initialized_)
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

    U = new float[2*N*N];


    S = new float[3*N*N];
    S0 = new float[3*N*N];

    F = new float[2*N*N];

    resetFluid();

    is_initialized_ = true;
}

JFS_INLINE void LBMSolver::setDensityVisBounds(float minrho, float maxrho)
{
    minrho_ = minrho;
    maxrho_ = maxrho;
}

JFS_INLINE void LBMSolver::getCurrentDensityBounds(float minmax_rho[2])
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

JFS_INLINE void LBMSolver::enableDensityViewMode(bool use)
{
    view_density_ = use;
}

JFS_INLINE void LBMSolver::resetFluid()
{

    for (int i = 0; i < N*N; i++)
    {
        U[2*i + 0] = 0;
        U[2*i + 1] = 0;

        S[2*i + 0] = 0;
        S[2*i + 1] = 0;
        S[2*i + 2] = 0;

        S0[3*i + 0] = 0;
        S0[3*i + 1] = 0;
        S0[3*i + 2] = 0;

        rho_[i] = rho0;
    }

    // reset distribution
    for (int j=0; j < N; j++)
        for (int k=0; k < N; k++)
            for (int i=0; i < 9; i++)
                f[N*9*k + 9*j + i] = calc_fbari(i, j, k);

    T = 0;
}

JFS_INLINE void LBMSolver::getImage(Eigen::VectorXf &image)
{
    if (view_density_)
        getDensityImage(image);
    else
        getSourceImage(image);
}

JFS_INLINE void LBMSolver::getSourceImage(Eigen::VectorXf &image)
{
    using grid2D = grid2D<Eigen::ColMajor>;
    
    auto btype = grid2D::bound_type_;
    auto L = grid2D::L;
    auto N = grid2D::N;
    auto D = grid2D::D;

    std::memcpy(image.data(), this->S, 3*N*N*sizeof(float));
}

JFS_INLINE void LBMSolver::getDensityImage(Eigen::VectorXf &image)
{
    using grid2D = grid2D<Eigen::ColMajor>;
    
    auto btype = grid2D::bound_type_;
    auto L = grid2D::L;
    auto N = grid2D::N;
    auto D = grid2D::D;

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

    if (image.rows() != N*N*3)
        image.resize(N*N*3);

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

            int map_idx = (int) (rho * 255);

            map_idx = (map_idx > 255) ? 255 : map_idx;
            map_idx = (map_idx < 0) ? 0 : map_idx;

            image[N*3*j + 0 + i*3] = (float) map_idx / 255;
            image[N*3*j + 1 + i*3] = (float) map_idx / 255;
            image[N*3*j + 2 + i*3] = (float) map_idx / 255;
        }
}

JFS_INLINE void LBMSolver::forceVelocity(int i, int j, float ux, float uy)
{

            Eigen::VectorXi indices(2);
            indices(0) = i;
            indices(1) = j;

            Vector_ u(2);
            u(0) = ux;
            u(1) = uy;

            insertIntoGrid(indices.data(), u.data(), U, VECTOR_FIELD, 1);

            Vector_ fbar(9);
            for (int k = 0; k < 9; k++)
            {
                fbar(k) = calc_fbari(k, i, j);
            }

            insertIntoGrid(indices.data(), fbar.data(), f, SCALAR_FIELD, 9);

            calcPhysicalVals(i, j);
}

JFS_INLINE void LBMSolver::forceDensity(int i, int j, float rho)
{

            Eigen::VectorXi indices(2);
            indices(0) = i;
            indices(1) = j;

            insertIntoGrid(indices.data(), &rho, rho_, SCALAR_FIELD, 1);

            Vector_ fbar(9);
            for (int k = 0; k < 9; k++)
            {
                fbar(k) = calc_fbari(k, i, j);
            }

            insertIntoGrid(indices.data(), fbar.data(), f, SCALAR_FIELD, 9);

            calcPhysicalVals(i, j);
}

JFS_INLINE void LBMSolver::doPressureWave(PressureWave p_wave)
{
    int i = p_wave.x[0]/dx;
    if (i < 0)
        i == 0;
    if (i >= N)
        i = N-1;
    int j = p_wave.x[1]/dx;
    if (j < 0)
        j == 0;
    if (j >= N)
        j = N-1;

    int radius = p_wave.radius/dx;

    int j_min = j - p_wave.radius;
    if (j_min < 0)
        j_min == 0;
    if (j_min >= N)
        j_min = N-1;
    int j_max = j + p_wave.radius;
    if (j_max < 0)
        j_max == 0;
    if (j_max >= N)
        j_max = N-1;

    for (j = j_min; j <= j_max; j++)
    {
        float Hz = 1.e3;
        float w = M_PI * Hz;
        float ux = p_wave.u_imp * (std::sin(w * T) + 1);
        forceVelocity(i, j, ux, 0);
    }

    // int i = (p_wave.x[0] + p_wave.u[0] * (T - p_wave.t_start)) / dx;
    // int j = (p_wave.x[1] + p_wave.u[1] * (T - p_wave.t_start)) / dx;

    // float r_real = p_wave.radius;
    // float u_imp = p_wave.u_imp;

    // float t_start = p_wave.t_start;

    // float w = u_imp / r_real;
    // float Hz = w / (2 * M_PI);
    // float period = 1/Hz;
    // u_imp *= r_real * std::cos(w * (T-t_start));

    // if ( (T-t_start) > period/2 )
    //     return;

    // int r = (r_real * std::sin(w * (T-t_start)))/dx;
    
    // int x = r, y = 0; 

    // Eigen::VectorXi indices(2);
    // indices(0) = i + x;
    // indices(1) = j + y;
    
    // Vector_ dir(2);
    // dir(0) = (float) indices(0) - i;
    // dir(1) = (float) indices(1) - j;
    // dir.normalize();

    // float ux = dir(0) * u_imp + p_wave.u(0);
    // float uy = dir(1) * u_imp + p_wave.u(1);

    // bool speed_check;
    // bool idx_check;

    // idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    // speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    // if ( (idx_check && speed_check) || p_wave.skadoosh )
    //     forceVelocity(indices(0),indices(1), ux, uy);
      
    // // When radius is zero only a single 
    // // point will be printed 
    // if (r > 0) 
    // { 
    //     // 1
    //     indices(0) = i + x;
    //     indices(1) = j - y;
        
    //     dir(0) = (float) indices(0) - i;
    //     dir(1) = (float) indices(1) - j;
    //     dir.normalize();

    //     ux = dir(0) * u_imp + p_wave.u(0);
    //     uy = dir(1) * u_imp + p_wave.u(1);

    //     idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    //     speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    //     if ( (idx_check && speed_check) || p_wave.skadoosh )
    //         forceVelocity(indices(0),indices(1), ux, uy);

    //     // 2
    //     indices(0) = i + y;
    //     indices(1) = j + x;
        
    //     dir(0) = (float) indices(0) - i;
    //     dir(1) = (float) indices(1) - j;
    //     dir.normalize();

    //     ux = dir(0) * u_imp + p_wave.u(0);
    //     uy = dir(1) * u_imp + p_wave.u(1);

    //     idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    //     speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    //     if ( (idx_check && speed_check) || p_wave.skadoosh )
    //         forceVelocity(indices(0),indices(1), ux, uy);

    //     // 3
    //     indices(0) = i - y;
    //     indices(1) = j + x;
        
    //     dir(0) = (float) indices(0) - i;
    //     dir(1) = (float) indices(1) - j;
    //     dir.normalize();

    //     ux = dir(0) * u_imp + p_wave.u(0);
    //     uy = dir(1) * u_imp + p_wave.u(1);

    //     idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    //     speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    //     if ( (idx_check && speed_check) || p_wave.skadoosh )
    //         forceVelocity(indices(0),indices(1), ux, uy);
    // } 
      
    // // Initialising the value of P 
    // int P = 1 - r; 
    // while (x > y) 
    // {  
    //     y++; 
          
    //     // Mid-point is inside or on the perimeter 
    //     if (P <= 0) 
    //         P = P + 2*y + 1; 
              
    //     // Mid-point is outside the perimeter 
    //     else
    //     { 
    //         x--; 
    //         P = P + 2*y - 2*x + 1; 
    //     } 
          
    //     // All the perimeter points have already been printed 
    //     if (x < y) 
    //         break; 
          
    //     // Printing the generated point and its reflection 
    //     // in the other octants after translation 
    //     // 1 
    //     indices(0) = i + x;
    //     indices(1) = j + y;
        
    //     dir(0) = (float) indices(0) - i;
    //     dir(1) = (float) indices(1) - j;
    //     dir.normalize();
        
    //     ux = dir(0) * u_imp + p_wave.u(0);
    //     uy = dir(1) * u_imp + p_wave.u(1);

    //     idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    //     speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    //     if ( (idx_check && speed_check) || p_wave.skadoosh )
    //         forceVelocity(indices(0),indices(1), ux, uy);

    //     // 2 
    //     indices(0) = i - x;
    //     indices(1) = j + y;
        
    //     dir(0) = (float) indices(0) - i;
    //     dir(1) = (float) indices(1) - j;
    //     dir.normalize();

    //     ux = dir(0) * u_imp + p_wave.u(0);
    //     uy = dir(1) * u_imp + p_wave.u(1);

    //     idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    //     speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    //     if ( (idx_check && speed_check) || p_wave.skadoosh )
    //         forceVelocity(indices(0),indices(1), ux, uy);

    //     // 3 
    //     indices(0) = i + x;
    //     indices(1) = j - y;
        
    //     dir(0) = (float) indices(0) - i;
    //     dir(1) = (float) indices(1) - j;
    //     dir.normalize();

    //     ux = dir(0) * u_imp + p_wave.u(0);
    //     uy = dir(1) * u_imp + p_wave.u(1);

    //     idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    //     speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    //     if ( (idx_check && speed_check) || p_wave.skadoosh )
    //         forceVelocity(indices(0),indices(1), ux, uy);

    //     // 4 
    //     indices(0) = i - x;
    //     indices(1) = j - y;
        
    //     dir(0) = (float) indices(0) - i;
    //     dir(1) = (float) indices(1) - j;
    //     dir.normalize();

    //     ux = dir(0) * u_imp + p_wave.u(0);
    //     uy = dir(1) * u_imp + p_wave.u(1);

    //     idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    //     speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    //     if ( (idx_check && speed_check) || p_wave.skadoosh )
    //         forceVelocity(indices(0),indices(1), ux, uy);
          
    //     // If the generated point is on the line x = y then  
    //     // the perimeter points have already been printed 
    //     if (x != y) 
    //     { 
    //         // 1 
    //         indices(0) = i + y;
    //         indices(1) = j + x;
            
    //         dir(0) = (float) indices(0) - i;
    //         dir(1) = (float) indices(1) - j;
    //         dir.normalize();

    //         ux = dir(0) * u_imp + p_wave.u(0);
    //         uy = dir(1) * u_imp + p_wave.u(1);

    //         idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    //         speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    //         if ( (idx_check && speed_check) || p_wave.skadoosh )
    //             forceVelocity(indices(0),indices(1), ux, uy);

    //         // 2 
    //         indices(0) = i - y;
    //         indices(1) = j + x;
            
    //         dir(0) = (float) indices(0) - i;
    //         dir(1) = (float) indices(1) - j;
    //         dir.normalize();

    //         ux = dir(0) * u_imp + p_wave.u(0);
    //         uy = dir(1) * u_imp + p_wave.u(1);

    //         idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    //         speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    //         if ( (idx_check && speed_check) || p_wave.skadoosh )
    //             forceVelocity(indices(0),indices(1), ux, uy);

    //         // 3 
    //         indices(0) = i + y;
    //         indices(1) = j - x;
            
    //         dir(0) = (float) indices(0) - i;
    //         dir(1) = (float) indices(1) - j;
    //         dir.normalize();

    //         ux = dir(0) * u_imp + p_wave.u(0);
    //         uy = dir(1) * u_imp + p_wave.u(1);

    //         idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    //         speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    //         if ( (idx_check && speed_check) || p_wave.skadoosh )
    //             forceVelocity(indices(0),indices(1), ux, uy);

    //         // 4 
    //         indices(0) = i - y;
    //         indices(1) = j - x;
            
    //         dir(0) = (float) indices(0) - i;
    //         dir(1) = (float) indices(1) - j;
    //         dir.normalize();

    //         ux = dir(0) * u_imp + p_wave.u(0);
    //         uy = dir(1) * u_imp + p_wave.u(1);

    //         idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    //         speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    //         if ( (idx_check && speed_check) || p_wave.skadoosh )
    //             forceVelocity(indices(0),indices(1), ux, uy);
    //     } 
    // } 
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
            Eigen::VectorXi indices(2);
            indices(0) = i + step;
            indices(1) = j;
            Vector_ u(2);
            indexGrid(u.data(), indices.data(), U, VECTOR_FIELD);
            Vector_ rho(1);
            indexGrid(rho.data(), indices.data(), rho_, SCALAR_FIELD);

            indices(0) = i;

            insertIntoGrid(indices.data(), rho.data(), rho_, SCALAR_FIELD);
            forceVelocity(indices(0), indices(1), u(0), u(1));
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
            Eigen::VectorXi indices(2);
            indices(1) = i + step;
            indices(0) = j;
            Vector_ u(2);
            indexGrid(u.data(), indices.data(), U, VECTOR_FIELD);
            Vector_ rho(1);
            indexGrid(rho.data(), indices.data(), rho_, SCALAR_FIELD);

            indices(1) = i;

            insertIntoGrid(indices.data(), rho.data(), rho_, SCALAR_FIELD);
            forceVelocity(indices(0), indices(1), u(0), u(1));
        }
    }
}

JFS_INLINE bool LBMSolver::calcNextStep(const std::vector<PressureWave> p_waves)
{

    using grid2D = grid2D<Eigen::ColMajor>;

    BoundType btype = grid2D::bound_type_;

    int iterations = iter_per_frame;
    for (int iter = 0; iter < iterations; iter++)
    {

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
                        switch (i)
                        {
                        case 1:
                            fiStar = f0[N*9*k + 9*j + 2];
                            break;
                        case 2:
                            fiStar = f0[N*9*k + 9*j + 1];
                            break;
                        case 3:
                            fiStar = f0[N*9*k + 9*j + 4];
                            break;
                        case 4:
                            fiStar = f0[N*9*k + 9*j + 3];
                            break;
                        case 5:
                            fiStar = f0[N*9*k + 9*j + 8];
                            break;
                        case 6:
                            fiStar = f0[N*9*k + 9*j + 7];
                            break;
                        case 7:
                            fiStar = f0[N*9*k + 9*j + 6];
                            break;
                        case 8:
                            fiStar = f0[N*9*k + 9*j + 5];
                            break;
                        }
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
        for (int i = 0; i < p_waves.size(); i++)
            doPressureWave(p_waves[i]);
        
        // collide
        #pragma omp parallel
        #pragma omp for 
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

        backstream(S, S0, U, dt, SCALAR_FIELD, 3);
        std::memcpy(S0, S, 3*N*N*sizeof(float));

        T += dt;
    }

    return false;
}

JFS_INLINE bool LBMSolver::calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources)
{

    std::vector<PressureWave> p_waves;

    return calcNextStep(p_waves);
}

JFS_INLINE bool LBMSolver::calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources, const std::vector<PressureWave> p_waves)
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
        for (int i = 0; i < sources.size(); i++)
        {
            float color[3] = {
                sources[i].color[0] * sources[i].strength * dt,
                sources[i].color[1] * sources[i].strength * dt,
                sources[i].color[2] * sources[i].strength * dt
            };
            float point[3] = {
                sources[i].pos[0]/this->D,
                sources[i].pos[1]/this->D,
                sources[i].pos[2]/this->D
            };
            this->interpPointToGrid(color, point, S0, SCALAR_FIELD, 3, Add);
        }

        failedStep = calcNextStep(p_waves);

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
    delete [] f;
    delete [] f0;
    delete [] rho_;
    delete [] U;
    delete [] S;
    delete [] S0;
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

JFS_INLINE void LBMSolver::adj_uref()
{
    // float umaxL = 0;
    // float umax = 0;
    // Eigen::Vector2f u;
    // for (int j=0; j<N; j++)
    //     for (int k=0; k<N; k++)
    //     {
    //         u = {U(N*N*0 + N*k + j), U(N*N*1 + N*k + j)};
    //         if ( urefL/uref * u.norm() > umaxL)
    //         {
    //             umax = u.norm();
    //             umaxL = urefL/uref * umax;
    //         }
    //     }

    // std::cout << "umax: " << umax << std::endl;
    // std::cout << "1: " << uref << std::endl;

    // // keep stability and don't make time step too small
    // if (umaxL < .018 || umaxL > .022)
    //     uref = 5*urefL*umax;

    // std::cout << "2: " << uref << std::endl;

    // // stop uref from being set to zero
    // float epsilon = 1e-2;
    // if (uref < epsilon)
    //     uref = epsilon;
    
    // std::cout << "3: " << uref << std::endl;

    // // dont overshoot frame delta t ( can only make time step smaller, so no need to worry about stability )
    // float dtFrame = 1/fps;
    // float dtIter = urefL / uref * dx * dtL;
    // if (dt + dtIter > dtFrame)
    //     uref = urefL/(dtFrame-dt) * dx * dtL;
    
    // std::cout << "4: " << uref << std::endl;
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