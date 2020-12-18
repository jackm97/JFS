#ifndef FLUID2D_H
#define FLUID2D_H
#include "../jfs_inline.h"

#include <jfs/base/fluidBase.h>
#include <jfs/base/grid2D.h>

namespace jfs {

class fluid2D : public grid2D, public fluidBase {
    public:
        fluid2D(){}

        virtual void resetFluid();

        virtual void getImage(Eigen::VectorXf &img);

        ~fluid2D(){}

    protected:
        
        virtual void initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dt);

        virtual void interpolateForce(const std::vector<Force> forces);
        
        virtual void interpolateSource( const std::vector<Source> sources);
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/base/fluid2D.cpp>
#endif

#endif