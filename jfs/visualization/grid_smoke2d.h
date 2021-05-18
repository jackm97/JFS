#ifndef GRID_SMOKE2D_H
#define GRID_SMOKE2D_H
#include "../jfs_inline.h"

#include <jfs/grid/grid2D.h>

#include <vector>

namespace jfs {

class gridSmoke2D : public grid2D {
    public:
        gridSmoke2D(){}

        gridSmoke2D(unsigned int N, float L, BoundType btype, float dt, float diss=0);

        void initialize(unsigned int N, float L, BoundType btype, float dt, float diss=0);

        void resetSmoke();

        void updateSmoke(std::vector<Source> sources, float* u_field);

        //inline getters
        float* smokeData(){return S_;}

        ~gridSmoke2D(){ clearGrid(); }

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
#   include <jfs/visualization/grid_2d.cpp>
#endif

#endif