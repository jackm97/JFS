#include <jfs/visualization/grid_smoke2d.h>

#include <cstring>

namespace jfs {

JFS_INLINE gridSmoke2D::gridSmoke2D(unsigned int N, float L, BoundType btype, float dt, float diss)
{
    initialize(N, L, btype, dt, diss);
}

JFS_INLINE void gridSmoke2D::initialize(unsigned int N, float L, BoundType btype, float dt, float diss)
{
    clearGrid();

    initializeGrid(N, L, btype, dt);

    diss_ = diss;

    S_ = new float[3*N*N];
    S0_ = new float[3*N*N];

    resetSmoke();

    is_init_ = true;
}

JFS_INLINE void gridSmoke2D::resetSmoke()
{
    for (int i = 0; i < (3*N*N); i++)
    {
        S_[i] = 0;
        S0_[i] = 0;
    }
}

JFS_INLINE void gridSmoke2D::dissipate()
{
    for (int i = 0; i < (3*N*N); i++)
    {
        S_[i] = S0_[i] / (1 + dt * diss_);
    }
}

} // namespace jfs