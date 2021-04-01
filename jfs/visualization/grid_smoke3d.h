#ifndef GRID_SMOKE3D_H
#define GRID_SMOKE3D_H
#include "../jfs_inline.h"

#include <jfs/visualization/grid_smoke_base.h>
#include <jfs/base/grid3D.h>

namespace jfs {

class gridSmoke3D : public virtual gridSmokeBase, public virtual grid3D {
    public:
        gridSmoke3D(){}

        gridSmoke3D(unsigned int N, float L, BoundType btype, float dt, float diss=0);

        virtual void initialize(unsigned int N, float L, BoundType btype, float dt, float diss=0);

        virtual void resetSmoke();

        ~gridSmoke3D(){ clearGrid(); }

    protected:
        virtual void dissipate();
};

} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/visualization/grid_3d.cpp>
#endif

#endif