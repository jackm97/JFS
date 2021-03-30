#ifndef GRID_SMOKE_H
#define GRID_SMOKE_H
#include "../jfs_inline.h"

#include <jfs/base/gridBase.h>

#include <vector>

namespace jfs {

class gridSmokeBase : virtual public gridBase {
    public:
        gridSmokeBase(){}

        virtual void initialize(unsigned int N, float L, BoundType btype, float dt, float diss=0) = 0;

        virtual void resetSmoke() = 0;

        void updateSmoke(std::vector<Source> sources, float* u_field);

        //inline getters
        float* smokeData(){return S_;}

        ~gridSmokeBase(){ clearGrid(); }

    protected:
        bool is_init_ = false;

        float* S_;
        float* S0_;

        float diss_;

        virtual void dissipate() = 0;

        void clearGrid();
};

} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/visualization/grid_smoke_base.cpp>
#endif


#endif