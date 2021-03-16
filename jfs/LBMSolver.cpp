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
    
    // lattice scaling stuff
    cs = 1/std::sqrt(3);
    us = cs/urefL * uref;

    // dummy dt because it is calculated
    float dummy_dt;

    initializeGrid(N, L, btype, dummy_dt);
    
    viscL = urefL/(uref * dx) * visc;
    tau = (3*viscL + .5);
    dt = urefL/uref * dx * dtL;

    U.resize(N*N*2);
    
    S.resize(3*N*N);
    SF.resize(3*N*N);

    F.resize(N*N*2);

    f.resize(9*N*N);
    rho.resize(N*N);

    resetFluid();
}

JFS_INLINE void LBMSolver::setDensityVisBounds(float minrho, float maxrho)
{
    this->minrho = minrho;
    this->maxrho = maxrho;
}

JFS_INLINE void LBMSolver::getCurrentDensityBounds(float minmax_rho[2])
{

    float minrho_ = rho(0);
    float maxrho_ = rho(0);

    for (int i=0; i < N*N; i++)
    {
        if (rho(i) < minrho_)
            minrho_ = rho(i);
    }

    for (int i=0; i < N*N; i++)
    {
        if (rho(i) > maxrho_)
            maxrho_ = rho(i);
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
    U.setZero();

    S.setZero();
    S0 = S;
    SF.setZero();

    rho.setOnes();
    rho *= rho0;

    // reset distribution
    for (int j=0; j < N; j++)
        for (int k=0; k < N; k++)
            for (int i=0; i < 9; i++)
                f(N*N*i + N*k + j) = calc_fbari(i, j, k);

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

    if (image.rows() != N*N*3)
        image.resize(N*N*3);

    for (int i=0; i < N; i++)
        for (int j=0; j < N; j++)
        {
            image(N*3*j + 0 + i*3) = this->S(0*N*N + N*j + i);
            image(N*3*j + 1 + i*3) = this->S(1*N*N + N*j + i);
            image(N*3*j + 2 + i*3) = this->S(2*N*N + N*j + i);
        }
    image = (image.array() <= 1.).select(image, 1.);
}

JFS_INLINE void LBMSolver::getDensityImage(Eigen::VectorXf &image)
{
    using grid2D = grid2D<Eigen::ColMajor>;
    
    auto btype = grid2D::bound_type_;
    auto L = grid2D::L;
    auto N = grid2D::N;
    auto D = grid2D::D;

    float minrho_ = rho(0);
    float maxrho_ = rho(0);
    float meanrho_ = rho.mean();

    for (int i=0; i < N*N && minrho == -1; i++)
    {
        if (rho(i) < minrho_)
            minrho_ = rho(i);
    }

    if (minrho != -1)
        minrho_ = minrho;

    for (int i=0; i < N*N && maxrho == -1; i++)
    {
        if (rho(i) > maxrho_)
            maxrho_ = rho(i);
    }

    if (maxrho == -1 && minrho == -1)
    {
        if (maxrho_ - meanrho_ > meanrho_ - minrho_)
            minrho_ = meanrho_ - (maxrho_ - meanrho_);
        else
            maxrho_ = meanrho_ - (minrho_ - meanrho_);
    }

    if (maxrho != -1)
        maxrho_ = maxrho;

    if (image.rows() != N*N*3)
        image.resize(N*N*3);

    for (int i=0; i < N; i++)
        for (int j=0; j < N; j++)
        {
            Eigen::VectorXi indices(2);
            indices(0) = i;
            indices(1) = j;
            Vector_ rho_ = indexField(indices, rho, SCALAR_FIELD, 1);
            if ((maxrho_ - minrho_) != 0)
                rho_ = (rho_.array() - minrho_)/(maxrho_ - minrho_);
            else
                rho_ = 0 * rho_;

            int map_idx = (int) (rho_(0) * 255);

            map_idx = (map_idx > 255) ? 255 : map_idx;
            map_idx = (map_idx < 0) ? 0 : map_idx;

            image(N*3*j + 0 + i*3) = (float) map_idx / 255;
            image(N*3*j + 1 + i*3) = (float) map_idx / 255;
            image(N*3*j + 2 + i*3) = (float) map_idx / 255;
        }
    image = (image.array() <= 1.).select(image, 1.);
}

JFS_INLINE void LBMSolver::forceVelocity(int i, int j, float ux, float uy)
{

            Eigen::VectorXi indices(2);
            indices(0) = i;
            indices(1) = j;

            Vector_ u(2);
            u(0) = ux;
            u(1) = uy;

            insertIntoField(indices, u, U, VECTOR_FIELD, 1);

            Vector_ fbar(9);
            for (int k = 0; k < 9; k++)
            {
                fbar(k) = calc_fbari(k, i, j);
            }

            insertIntoField(indices, fbar, f, SCALAR_FIELD, 9);

            calcPhysicalVals(i, j);
}

JFS_INLINE void LBMSolver::doPressureWave(PressureWave p_wave)
{

    int i = (p_wave.x[0] + p_wave.u[0] * (T - p_wave.t_start)) / dx;
    int j = (p_wave.x[1] + p_wave.u[1] * (T - p_wave.t_start)) / dx;

    float r_real = p_wave.radius;
    float u_imp = p_wave.u_imp;

    float t_start = p_wave.t_start;

    float w = u_imp / r_real;
    float Hz = w / (2 * M_PI);
    float period = 1/Hz;
    u_imp *= r_real * std::cos(w * (T-t_start));

    if ( (T-t_start) > period/2 )
        return;

    int r = (r_real * std::sin(w * (T-t_start)))/dx;
    
    int x = r, y = 0; 

    Eigen::VectorXi indices(2);
    indices(0) = i + x;
    indices(1) = j + y;
    
    Vector_ dir(2);
    dir(0) = (float) indices(0) - i;
    dir(1) = (float) indices(1) - j;
    dir.normalize();

    float ux = dir(0) * u_imp + p_wave.u(0);
    float uy = dir(1) * u_imp + p_wave.u(1);

    bool speed_check;
    bool idx_check;

    idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
    speed_check = (ux * dir(0) + uy * dir(1)) > 0;
    if ( (idx_check && speed_check) || p_wave.skadoosh )
        forceVelocity(indices(0),indices(1), ux, uy);
      
    // When radius is zero only a single 
    // point will be printed 
    if (r > 0) 
    { 
        // 1
        indices(0) = i + x;
        indices(1) = j - y;
        
        dir(0) = (float) indices(0) - i;
        dir(1) = (float) indices(1) - j;
        dir.normalize();

        ux = dir(0) * u_imp + p_wave.u(0);
        uy = dir(1) * u_imp + p_wave.u(1);

        idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
        speed_check = (ux * dir(0) + uy * dir(1)) > 0;
        if ( (idx_check && speed_check) || p_wave.skadoosh )
            forceVelocity(indices(0),indices(1), ux, uy);

        // 2
        indices(0) = i + y;
        indices(1) = j + x;
        
        dir(0) = (float) indices(0) - i;
        dir(1) = (float) indices(1) - j;
        dir.normalize();

        ux = dir(0) * u_imp + p_wave.u(0);
        uy = dir(1) * u_imp + p_wave.u(1);

        idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
        speed_check = (ux * dir(0) + uy * dir(1)) > 0;
        if ( (idx_check && speed_check) || p_wave.skadoosh )
            forceVelocity(indices(0),indices(1), ux, uy);

        // 3
        indices(0) = i - y;
        indices(1) = j + x;
        
        dir(0) = (float) indices(0) - i;
        dir(1) = (float) indices(1) - j;
        dir.normalize();

        ux = dir(0) * u_imp + p_wave.u(0);
        uy = dir(1) * u_imp + p_wave.u(1);

        idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
        speed_check = (ux * dir(0) + uy * dir(1)) > 0;
        if ( (idx_check && speed_check) || p_wave.skadoosh )
            forceVelocity(indices(0),indices(1), ux, uy);
    } 
      
    // Initialising the value of P 
    int P = 1 - r; 
    while (x > y) 
    {  
        y++; 
          
        // Mid-point is inside or on the perimeter 
        if (P <= 0) 
            P = P + 2*y + 1; 
              
        // Mid-point is outside the perimeter 
        else
        { 
            x--; 
            P = P + 2*y - 2*x + 1; 
        } 
          
        // All the perimeter points have already been printed 
        if (x < y) 
            break; 
          
        // Printing the generated point and its reflection 
        // in the other octants after translation 
        // 1 
        indices(0) = i + x;
        indices(1) = j + y;
        
        dir(0) = (float) indices(0) - i;
        dir(1) = (float) indices(1) - j;
        dir.normalize();
        
        ux = dir(0) * u_imp + p_wave.u(0);
        uy = dir(1) * u_imp + p_wave.u(1);

        idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
        speed_check = (ux * dir(0) + uy * dir(1)) > 0;
        if ( (idx_check && speed_check) || p_wave.skadoosh )
            forceVelocity(indices(0),indices(1), ux, uy);

        // 2 
        indices(0) = i - x;
        indices(1) = j + y;
        
        dir(0) = (float) indices(0) - i;
        dir(1) = (float) indices(1) - j;
        dir.normalize();

        ux = dir(0) * u_imp + p_wave.u(0);
        uy = dir(1) * u_imp + p_wave.u(1);

        idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
        speed_check = (ux * dir(0) + uy * dir(1)) > 0;
        if ( (idx_check && speed_check) || p_wave.skadoosh )
            forceVelocity(indices(0),indices(1), ux, uy);

        // 3 
        indices(0) = i + x;
        indices(1) = j - y;
        
        dir(0) = (float) indices(0) - i;
        dir(1) = (float) indices(1) - j;
        dir.normalize();

        ux = dir(0) * u_imp + p_wave.u(0);
        uy = dir(1) * u_imp + p_wave.u(1);

        idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
        speed_check = (ux * dir(0) + uy * dir(1)) > 0;
        if ( (idx_check && speed_check) || p_wave.skadoosh )
            forceVelocity(indices(0),indices(1), ux, uy);

        // 4 
        indices(0) = i - x;
        indices(1) = j - y;
        
        dir(0) = (float) indices(0) - i;
        dir(1) = (float) indices(1) - j;
        dir.normalize();

        ux = dir(0) * u_imp + p_wave.u(0);
        uy = dir(1) * u_imp + p_wave.u(1);

        idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
        speed_check = (ux * dir(0) + uy * dir(1)) > 0;
        if ( (idx_check && speed_check) || p_wave.skadoosh )
            forceVelocity(indices(0),indices(1), ux, uy);
          
        // If the generated point is on the line x = y then  
        // the perimeter points have already been printed 
        if (x != y) 
        { 
            // 1 
            indices(0) = i + y;
            indices(1) = j + x;
            
            dir(0) = (float) indices(0) - i;
            dir(1) = (float) indices(1) - j;
            dir.normalize();

            ux = dir(0) * u_imp + p_wave.u(0);
            uy = dir(1) * u_imp + p_wave.u(1);

            idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
            speed_check = (ux * dir(0) + uy * dir(1)) > 0;
            if ( (idx_check && speed_check) || p_wave.skadoosh )
                forceVelocity(indices(0),indices(1), ux, uy);

            // 2 
            indices(0) = i - y;
            indices(1) = j + x;
            
            dir(0) = (float) indices(0) - i;
            dir(1) = (float) indices(1) - j;
            dir.normalize();

            ux = dir(0) * u_imp + p_wave.u(0);
            uy = dir(1) * u_imp + p_wave.u(1);

            idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
            speed_check = (ux * dir(0) + uy * dir(1)) > 0;
            if ( (idx_check && speed_check) || p_wave.skadoosh )
                forceVelocity(indices(0),indices(1), ux, uy);

            // 3 
            indices(0) = i + y;
            indices(1) = j - x;
            
            dir(0) = (float) indices(0) - i;
            dir(1) = (float) indices(1) - j;
            dir.normalize();

            ux = dir(0) * u_imp + p_wave.u(0);
            uy = dir(1) * u_imp + p_wave.u(1);

            idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
            speed_check = (ux * dir(0) + uy * dir(1)) > 0;
            if ( (idx_check && speed_check) || p_wave.skadoosh )
                forceVelocity(indices(0),indices(1), ux, uy);

            // 4 
            indices(0) = i - y;
            indices(1) = j - x;
            
            dir(0) = (float) indices(0) - i;
            dir(1) = (float) indices(1) - j;
            dir.normalize();

            ux = dir(0) * u_imp + p_wave.u(0);
            uy = dir(1) * u_imp + p_wave.u(1);

            idx_check = !(indices(0) > N-1 || indices(0) < 0 || indices(1) > N-1 || indices(1) < 0);
            speed_check = (ux * dir(0) + uy * dir(1)) > 0;
            if ( (idx_check && speed_check) || p_wave.skadoosh )
                forceVelocity(indices(0),indices(1), ux, uy);
        } 
    } 
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
            Vector_ u_ = indexField(indices, U, VECTOR_FIELD);
            Vector_ rho_ = indexField(indices, rho, SCALAR_FIELD);

            indices(0) = i;

            insertIntoField(indices, rho_, rho, SCALAR_FIELD);
            forceVelocity(indices(0), indices(1), u_(0), u_(1));
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
            Vector_ u_ = indexField(indices, U, VECTOR_FIELD);
            Vector_ rho_ = indexField(indices, rho, SCALAR_FIELD);

            indices(1) = i;

            insertIntoField(indices, rho_, rho, SCALAR_FIELD);
            forceVelocity(indices(0), indices(1), u_(0), u_(1));
        }
    }
}

JFS_INLINE bool LBMSolver::calcNextStep(const std::vector<PressureWave> p_waves)
{

    using grid2D = grid2D<Eigen::ColMajor>;

    BoundType btype = grid2D::bound_type_;

    static Vector_ f0;

    int iterations = iter_per_frame;
    for (int iter = 0; iter < iterations; iter++)
    {

        f0 = f;
        
        // stream
        for (int idx=0; idx<(N*N*9); idx++)
        {
            float fiStar;
            int i = std::floor(((float)idx)/(N*N));
            int k = std::floor( ( (float)idx-N*N*i )/N );
            int j = std::floor( ( (float)idx-N*N*i-N*k ) );

                    int cix = c[i](0);
                    int ciy = c[i](1);

                    if ((k-ciy) >= 0 && (k-ciy) < N && (j-cix) >= 0 && (j-cix) < N)
                        fiStar = f0(N*N*i + N*(k-ciy) + (j-cix));
                    else
                    {
                        switch (i)
                        {
                        case 1:
                            fiStar = f0(N*N*2 + N*k + j);
                            break;
                        case 2:
                            fiStar = f0(N*N*1 + N*k + j);
                            break;
                        case 3:
                            fiStar = f0(N*N*4 + N*k + j);
                            break;
                        case 4:
                            fiStar = f0(N*N*3 + N*k + j);
                            break;
                        case 5:
                            fiStar = f0(N*N*8 + N*k + j);
                            break;
                        case 6:
                            fiStar = f0(N*N*7 + N*k + j);
                            break;
                        case 7:
                            fiStar = f0(N*N*6 + N*k + j);
                            break;
                        case 8:
                            fiStar = f0(N*N*5 + N*k + j);
                            break;
                        }
                    }

                    f(N*N*i + N*k + j) = fiStar; 

                    calcPhysicalVals(j, k);
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
            int i = std::floor(((float)idx)/(N*N));
            int k = std::floor( ( (float)idx-N*N*i )/N );
            int j = std::floor( ( (float)idx-N*N*i-N*k ) ); 
            
            fi = f(N*N*i + N*k + j);

            fbari = calc_fbari(i, j, k);            
                
            Fi = calc_Fi(i, j, k);

            Omegai = -(fi - fbari)/tau;

            f(N*N*i + N*k + j) = fi + Omegai + Fi;
        }

        bool badStep = Eigen::isinf(U.array()).any() || Eigen::isnan(U.array()).any();
        if (badStep) return true;

        addForce(S, S0, SF, dt);
        backstream(S0, S, U, dt, SCALAR_FIELD, 2);

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
        interpolateForce(forces, F);
        interpolateSource(sources, SF);

        failedStep = calcNextStep(p_waves);

        F.setZero();
        SF.setZero();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        failedStep = true;
    }

    if (failedStep) resetFluid();

    return failedStep;
}

JFS_INLINE void LBMSolver::addForce(Vector_ &dst, const Vector_ &src, const Vector_ &force, float dt)
{
    dst = src + dt * force ;
}

JFS_INLINE float LBMSolver::calc_fbari(int i, int j, int k)
{
    float fbari;
    float rhoP = rho(N*k + j); // rho at point P -> (j,k)
    float wi = w[i];

    Eigen::Vector2f u, ci;

    u = {U(N*N*0 + N*k + j), U(N*N*1 + N*k + j)};
    u = urefL/uref * u;
    ci = c[i];

    fbari = wi * rhoP/rho0 * ( 1 + ci.dot(u)/(std::pow(cs,2)) + std::pow(ci.dot(u),2)/(2*std::pow(cs,4)) - u.dot(u)/(2*std::pow(cs,2)) );

    return fbari;
}

JFS_INLINE float LBMSolver::calc_Fi(int i, int j, int k)
{
    float wi = w[i];

    Eigen::Vector2f u, ci, FP;

    u = {U(N*N*0 + N*k + j), U(N*N*1 + N*k + j)};
    u = urefL/uref * u;
    ci = c[i];
    FP = {F.coeff(N*N*0 + N*k + j), F.coeff(N*N*1 + N*k + j)};
    FP = ( 1/rho0 * dx * std::pow(urefL/uref,2) ) * FP;

    float Fi = (1 - tau/2) * wi *
                ( (1/std::pow(cs,2))*(ci - u) + (ci.dot(u)/std::pow(cs,4)) * ci ).dot(FP);

    return Fi;
}

JFS_INLINE void LBMSolver::adj_uref()
{
    float umaxL = 0;
    float umax = 0;
    Eigen::Vector2f u;
    for (int j=0; j<N; j++)
        for (int k=0; k<N; k++)
        {
            u = {U(N*N*0 + N*k + j), U(N*N*1 + N*k + j)};
            if ( urefL/uref * u.norm() > umaxL)
            {
                umax = u.norm();
                umaxL = urefL/uref * umax;
            }
        }

    std::cout << "umax: " << umax << std::endl;
    std::cout << "1: " << uref << std::endl;

    // keep stability and don't make time step too small
    if (umaxL < .018 || umaxL > .022)
        uref = 5*urefL*umax;

    std::cout << "2: " << uref << std::endl;

    // stop uref from being set to zero
    float epsilon = 1e-2;
    if (uref < epsilon)
        uref = epsilon;
    
    std::cout << "3: " << uref << std::endl;

    // dont overshoot frame delta t ( can only make time step smaller, so no need to worry about stability )
    float dtFrame = 1/fps;
    float dtIter = urefL / uref * dx * dtL;
    if (dt + dtIter > dtFrame)
        uref = urefL/(dtFrame-dt) * dx * dtL;
    
    std::cout << "4: " << uref << std::endl;
}

JFS_INLINE void LBMSolver::calcPhysicalVals()
{
    float rhoP;
    Eigen::Vector2f momentumP;

    for (int j=0; j<N; j++)
        for (int k=0; k<N; k++)
        {
            rhoP = 0;
            momentumP = {0, 0};
            for (int i=0; i<9; i++)
            {
                rhoP += f(N*N*i + N*k +j);
                momentumP += c[i] * f(N*N*i + N*k +j);
            }

            rho(N*k + j) = rho0 * rhoP;
            U(N*N*0 + N*k + j) = uref/urefL * (momentumP(0)/rhoP);
            U(N*N*1 + N*k + j) = uref/urefL * (momentumP(1)/rhoP);
        }
}

JFS_INLINE void LBMSolver::calcPhysicalVals(int j, int k)
{
    float rhoP;
    Eigen::Vector2f momentumP;

    rhoP = 0;
    momentumP = {0, 0};
    for (int i=0; i<9; i++)
    {
        rhoP += f(N*N*i + N*k +j);
        momentumP += c[i] * f(N*N*i + N*k +j);
    }

    rho(N*k + j) = rho0 * rhoP;
    U(N*N*0 + N*k + j) = uref/urefL * (momentumP(0)/rhoP);
    U(N*N*1 + N*k + j) = uref/urefL * (momentumP(1)/rhoP);
}

} // namespace jfs