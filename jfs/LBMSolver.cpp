#include <jfs/LBMSolver.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace jfs {

JFS_INLINE LBMSolver::LBMSolver(unsigned int N, float L, float fps, float rho0, float visc, float us)
{
    initialize(N, L, fps, rho0, visc, us);
}

JFS_INLINE void LBMSolver::initialize(unsigned int N, float L, float fps, float rho0, float visc, float us)
{
    this->fps = fps;
    this->rho0 = rho0;
    this->visc = visc;
    this->us = us;
    
    // lattice scaling stuff
    cs = 1/std::sqrt(3);
    uref = urefL/cs * us;

    // dummy dt because it is calculated
    float dummy_dt;
    initializeFluid(N, L, ZERO, dummy_dt);
}


JFS_INLINE void LBMSolver::initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dummy_dt)
{

    initializeGrid(N, L, BOUND, dummy_dt);
    
    viscL = urefL/(uref * dx) * visc;
    tau = (3*viscL + .5);
    this->dt = urefL/uref * dx * dtL;

    U.resize(N*N*2);
    
    S.resize(3*N*N);
    SF.resize(3*N*N);

    F.resize(N*N*2);

    f.resize(9*N*N);
    rho.resize(N*N);

    resetFluid();
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
    frame = 0;
}

JFS_INLINE void LBMSolver::initializeGrid(unsigned int N, float L, BOUND_TYPE BOUND, float dummy_dt)
{

    initializeGridProperties(N, L, BOUND, dummy_dt);

    // differential operators are not
    // used for this solver so the
    // memory associated with them
    // is minimized
    LAPLACE.resize(1,1);
    VEC_LAPLACE.resize(1,1);
    DIV.resize(1,1);
    GRAD.resize(1,1);
}

JFS_INLINE bool LBMSolver::calcNextStep()
{
    static Eigen::VectorXf f0;

    T += dt;

    while (T < 1/fps*(frame+1))
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
        
        // collide
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
        backstream(S0, S, U, dt, 1, 2);

        T += dt;
    }

    frame += 1;
    T -= dt; // that time step never run

    return false;
}

JFS_INLINE bool LBMSolver::calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources)
{

    bool failedStep = false;
    try
    {    
        interpolateForce(forces);
        interpolateSource(sources);

        failedStep = calcNextStep();

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

JFS_INLINE void LBMSolver::addForce(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &force, float dt)
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