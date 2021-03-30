#include <jfs/visualization/grid_smoke_base.h>

#include <cstring>

namespace jfs {

JFS_INLINE void gridSmokeBase::updateSmoke(std::vector<Source> sources, float* u_field)
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

JFS_INLINE void gridSmokeBase::clearGrid()
{
    if (!is_init_)
        return;
    
    delete [] S_;
    delete [] S0_;

    is_init_ = false;
}

} // namespace jfs