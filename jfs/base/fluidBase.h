#ifndef FLUIDBASE_H
#define FLUIDBASE_H
#include "../jfs_inline.h"

namespace jfs {

class fluidBase {
    public:
        fluidBase(){}
        
        virtual void resetFluid() = 0;

        virtual void getImage(Eigen::VectorXf &img) = 0;

        // returns true if the step failed
        virtual bool calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources) = 0;

        ~fluidBase(){}

    protected:
};
} // namespace jfs

#endif