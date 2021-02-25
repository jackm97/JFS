#ifndef FLUIDBASE_H
#define FLUIDBASE_H
#include "../jfs_inline.h"

namespace jfs {

class fluidBase {
    public:
        fluidBase(){}
        virtual void resetFluid(){}

        virtual void getImage(Eigen::VectorXf &img) = 0;

        // returns true if the step failed
        virtual bool calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources) = 0;

        ~fluidBase(){}

    protected:
        Eigen::VectorXf U; 
        Eigen::VectorXf U0;
        
        Eigen::VectorXf S;
        Eigen::VectorXf S0;

        Eigen::SparseVector<float> F;
        Eigen::SparseVector<float> SF;

        // returns true if the step failed
        virtual bool calcNextStep( ) = 0;
        
        virtual void initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dt) = 0;

        virtual void interpolateForce(const std::vector<Force> forces) = 0;
        
        virtual void interpolateSource( const std::vector<Source> sources) = 0;
};
} // namespace jfs

#endif