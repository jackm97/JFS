#ifndef GRID_SMOKE3D_H
#define GRID_SMOKE3D_H
#include "../jfs_inline.h"

#include <jfs/grid/grid3D.h>

#include <vector>

namespace jfs {

class gridSmoke3D : public grid3D {
    public:
        gridSmoke3D(){}

        gridSmoke3D(unsigned int N, float L, BoundType btype, float dt, float diss=0);

        void initialize(unsigned int N, float L, BoundType btype, float dt, float diss=0);

        void resetSmoke();

        void updateSmoke(std::vector<Source> sources, float* u_field);

        //inline getters
        float* smokeData(){return S_;}

        ~gridSmoke3D(){ clearGrid(); }

    protected:
        bool is_init_ = false;

        float* S_;
        float* S0_;

        float diss_;

        void dissipate();

        void clearGrid();
};

} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/visualization/grid_3d.cpp>
#endif

#endif