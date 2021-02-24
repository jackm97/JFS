#include <jfs/base/gridBase.h>
#include <iostream>

namespace jfs
{

void gridBase::initializeGridProperties(unsigned int N, float L, BOUND_TYPE BOUND, float dt)
{
    this->N = N;
    this->L = L;
    this->D = L/(N-1);
    this->BOUND = BOUND;
    this->dt = dt;
}

JFS_INLINE Eigen::VectorXf gridBase::sourceTrace(Eigen::VectorXf X, const Eigen::VectorXf &ufield, int dims, float dt)
{
    Eigen::VectorXi start_indices = ( X.array()/D - .5).template cast<int>();
    Eigen::VectorXf u = indexField(start_indices, ufield, dims);
    Eigen::VectorXf interp_indices = 1/D * (X + u*dt*.5).array() - .5;
    
    u = calcLinInterp(interp_indices, ufield, dims);
    X = X + dt * u;

    return X;
}

} // namespace jfs