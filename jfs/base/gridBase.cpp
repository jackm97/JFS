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

JFS_INLINE void gridBase::backstream(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &u, float dt, int dims)
{
    sourceTrace(X0, X, u, -dt);

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
    case 3:
        linInterpPtr = &linInterpVec;
        fields = 1;
        break;
    }

    calcLinInterp(*linInterpPtr, ij0, dims, fields);

    dst = *linInterpPtr * src;
}

JFS_INLINE void gridBase::sourceTrace(Eigen::VectorXf &X0, const Eigen::VectorXf &X, const Eigen::VectorXf &u, float dt)
{
    ij0 = 1/D * ( (X + 1/2 * (D * X)) + dt/2 * u ).array() - .5;
    
    calcLinInterp(linInterpVec, ij0, dim_type);
    X0 = X + dt * ( linInterpVec * u );
}

} // namespace jfs