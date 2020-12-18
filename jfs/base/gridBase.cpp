#include <jfs/base/gridBase.h>

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
} // namespace jfs