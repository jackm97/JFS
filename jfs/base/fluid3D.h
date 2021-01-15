#ifndef FLUID3D_H
#define FLUID3D_H
#include "../jfs_inline.h"

#include <jfs/base/fluidBase.h>
#include <jfs/base/grid3D.h>

namespace jfs {

class fluid3D : public grid3D, public fluidBase {
    public:
        fluid3D(){}

        virtual void resetFluid();

        virtual void getImage(Eigen::VectorXf &img);

        ~fluid3D(){}

    protected:
        
        virtual void initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dt);

        virtual void interpolateForce(const std::vector<Force> forces);
        
        virtual void interpolateSource( const std::vector<Source> sources);
};
} // namespace jfs

#ifndef JFS_STATIC
#   include <jfs/base/fluid3D.cpp>
#endif

#endif