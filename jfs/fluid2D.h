#ifndef FLUID2D_H
#define FLUID2D_H
#include "jfs_inline.h"

#include <jfs/fluidBase.h>
#include <jfs/grid2D.h>

namespace jfs {

class fluid2D : public grid2D, public fluidBase {
    protected:

        fluid2D(){}

        void interpolateForce(const std::vector<Force> forces);
        
        void interpolateSource( const std::vector<Source> sources);

        void initializeGrid(unsigned int N, float L, BOUND_TYPE BOUND, float dt);

        void resetFluid();

        ~fluid2D(){}
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/fluid2D.cpp>
#endif

#endif