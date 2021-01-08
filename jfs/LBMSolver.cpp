#include <jfs/LBMSolver.h>

#include <cmath>
#include <iostream>

namespace jfs {

JFS_INLINE LBMSolver::LBMSolver(unsigned int N, float L, float fps, float rho0, float visc, float us)
{
    initialize(N, L, fps, rho0, visc, us);
}

JFS_INLINE void LBMSolver::initialize(unsigned int N, float L, float fps, float rho0, float visc, float us)
{
    
    this->fps = fps;
    this->visc = visc;
    this->rho0 = rho0;
    this->us = us;

    // float dtFrame = 1/fps;
    // uref = urefL/dtFrame * dx * dtL;

    uref = .5*us;

    initializeFluid(N, L, ZERO, dt);
}


JFS_INLINE void LBMSolver::initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dt)
{

    initializeGrid(N, L, BOUND, dt);

    U.resize(N*N*2);
    Ub.resize(N*N*2);

    F.resize(N*N*2);
    Fb.resize(N*N*2);
    
    S.resize(3*N*N);
    SF.resize(3*N*N);

    f.resize(9*N*N);
    rho.resize(N*N);

    resetFluid();
}


JFS_INLINE void LBMSolver::resetFluid()
{
    U.setZero();
    for (int i=0; i < N; i++)
    {
        U(N*N*1 + N*(N-2) + i) = -.001 * us;
    }
    Ub.setZero();

    F.setZero();
    Fb.setZero();

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
}

JFS_INLINE void LBMSolver::initializeGrid(unsigned int N, float L, BOUND_TYPE BOUND, float dt)
{

    initializeGridProperties(N, L, BOUND, dt);

    X.resize(N*N*2);
    setXGrid();

    ij0.resize(N*N*2);
    linInterp.resize(N*N,N*N);
    linInterpVec.resize(N*N*2,N*N*2);
}

JFS_INLINE void LBMSolver::calcNextStep()
{
    static Eigen::VectorXf f0;
    static int count = 0;

    count +=1;

    f0 = f;

    float viscL = urefL/(uref * dx) * visc;
    float tau = 1/(3*viscL + .5);

    float fi;
    float fbari;
    float fiStar;
    float Omegai;
    float Fi;
    // adj_uref();
    addBoundaryForces(Ub, U, (urefL/uref * dx * dtL));
    for (int j=0; j<N; j++)
        for (int k=0; k<N; k++)
            for (int i=0; i<9; i++)
            {
                fi = f(N*N*i + N*k + j);

                int cix = c[i](0);
                int ciy = c[i](1);

                fbari = calc_fbari(i, j, k);

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
                
                    
                Fi = calc_Fi(i, j, k);

                Omegai = -(fi - fbari)/tau;

                f(N*N*i + N*k + j) = fiStar + Omegai + Fi;
                // std::cout << fiStar << std::endl;
                // std::cout << Omegai << std::endl;
                // std::cout << Fi << std::endl;
                // std::cout << fbari << std::endl;
                // std::cout << i << "," << j << "," << k << std::endl;
            }

    calcPhysicalVals();
    Ub = U;
    satisfyBC(Ub);
    // std::cout << (rho.array() < 0).sum() << std::endl;
    std::cout << (urefL/uref * U).array().abs().maxCoeff() << std::endl;
    // std::cout << F.toDense().array().abs().maxCoeff() << std::endl;
    // std::cout << Fb.toDense().array().abs().maxCoeff() << std::endl;
    // std::cout << U(N*N*0 + N*32 + 32) << "," <<  U(N*N*1 + N*32 + 32) << std::endl;

    // addForce(S, S0, SF, urefL/uref * dx * dtL);
    // transport(S0, S, U, urefL/uref * dx * dtL, 1);

    dt += urefL/uref * dx * dtL;

    if (dt >= 1/fps)
        dt = 0;
}

JFS_INLINE void LBMSolver::calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources)
{
    interpolateForce(forces);
    interpolateSource(sources);

    calcNextStep();

    F.setZero();
    SF.setZero();
}

JFS_INLINE void LBMSolver::addForce(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &force, float dt)
{
    dst = src + dt * force ;
}

JFS_INLINE void LBMSolver::transport(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt, int dims)
{
    particleTrace(X0, X, u, -dt);

    ij0 = (1/D * X0.array() - .5);

    SparseMatrix *linInterpPtr;
    int fields;

    switch (dims)
    {
    case 1:
        linInterpPtr = &linInterp;
        fields = 3;
        break;

    case 2:
        linInterpPtr = &linInterpVec;
        fields = 1;
        break;
    }

    calcLinInterp(*linInterpPtr, ij0, dims, fields);

    dst = *linInterpPtr * src;
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

    float cs = urefL/uref * us; // speed of sound in LBM units

    Eigen::Vector2f u, ci;

    u = {U(N*N*0 + N*k + j), U(N*N*1 + N*k + j)};
    u = urefL/uref * u;
    ci = c[i];

    fbari = wi * rhoP/rho0 * ( 1 + ci.dot(u)/(std::pow(cs,2)) + std::pow(ci.dot(u),2)/(2*std::pow(cs,4)) - u.dot(u)/(2*std::pow(cs,2)) );

    return fbari;
}

JFS_INLINE float LBMSolver::calc_Fi(int i, int j, int k)
{
    float viscL = urefL/(uref * dx) * visc;
    float tau = 1/(3*viscL + .5); // relaxation time
    float wi = w[i];

    float cs = urefL/uref * us; // speed of sound in LBM units

    Eigen::Vector2f u, ci, FP;

    u = {U(N*N*0 + N*k + j), U(N*N*1 + N*k + j)};
    u = urefL/uref * u;
    ci = c[i];
    FP = {F.coeff(N*N*0 + N*k + j) + Fb.coeff(N*N*0 + N*k + j), F.coeff(N*N*1 + N*k + j) + Fb.coeff(N*N*1 + N*k + j)};
    FP /= ( rho0/dx * std::pow(uref/urefL,2) );

    float Fi = (1 - 1/(2*tau)) * wi *
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
            U(N*N*0 + N*k + j) = urefL/uref * (momentumP(0)/rhoP);
            U(N*N*1 + N*k + j) = urefL/uref * (momentumP(1)/rhoP);
        }
}

void LBMSolver::addBoundaryForces(Eigen::VectorXf &Ub, Eigen::VectorXf &U, float dt)
{
    Fb.setZero();
    for (int j=0; j<N; j++)
        for (int k=0; k<N; k++)
        {
            float imp = rho(N*k +j) * (Ub(N*N*0 + N*k +j) - U(N*N*0 + N*k +j)) / dt;
            Fb.insert(N*N*0 + N*k + j) = imp;

            imp = rho(N*k +j) * (Ub(N*N*1 + N*k +j) - U(N*N*1 + N*k +j)) / dt;
            Fb.insert(N*N*1 + N*k + j) = imp;
        }
}

} // namespace jfs