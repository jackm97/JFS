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

JFS_INLINE void gridSmoke2D::updateSmoke(std::vector<Source> sources, float* u_field)
{
    for (int i = 0; i < sources.size(); i++)
    {
        float source[3] = {
            sources[i].color[0] * sources[i].strength,
            sources[i].color[1] * sources[i].strength,
            sources[i].color[2] * sources[i].strength
        };
        float point[3] = {
            sources[i].pos[0]/this->D,
            sources[i].pos[1]/this->D,
            sources[i].pos[2]/this->D
        };
        this->interpPointToGrid(source, point, S_, SCALAR_FIELD, 3, Add);
    }

    this->backstream(S0_, S_, u_field, dt, SCALAR_FIELD, 3);
    dissipate();
}

JFS_INLINE void gridSmoke2D::dissipate()
{
    for (int i = 0; i < (3*N*N); i++)
    {
        S_[i] = S0_[i] / (1 + dt * diss_);
    }
}

JFS_INLINE void gridSmoke2D::clearGrid()
{
    if (!is_init_)
        return;
    
    delete [] S_;
    delete [] S0_;

    is_init_ = false;
}

} // namespace jfs