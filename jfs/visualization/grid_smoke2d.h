#ifndef GRID_SMOKE2D_H
#define GRID_SMOKE2D_H
#include "../jfs_inline.h"

#include <jfs/visualization/grid_smoke_base.h>
#include <jfs/base/grid2D.h>

namespace jfs {

class gridSmoke2D : public virtual gridSmokeBase, public virtual grid2D {
    public:
        gridSmoke2D(){}

        gridSmoke2D(unsigned int N, float L, BoundType btype, float dt, float diss=0);

        virtual void initialize(unsigned int N, float L, BoundType btype, float dt, float diss=0);

        virtual void resetSmoke();

        ~gridSmoke2D(){ clearGrid(); }

    protected:
        virtual void dissipate();
};

} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/visualization/grid_2d.cpp>
#endif

#endif