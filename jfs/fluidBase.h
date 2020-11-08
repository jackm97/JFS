#ifndef FLUIDBASE_H
#define FLUIDBASE_H
#include "jfs_inline.h"

namespace jfs {

class fluidBase {
    public:
        virtual void initializeGrid(unsigned int N, float L, BOUND_TYPE BOUND, float dt){}

        virtual void resetFluid(){}

        virtual void getImage(Eigen::VectorXf img){}

        virtual void calcNextStep( ){}

        virtual void calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources){}

    protected:
        Eigen::VectorXf U; 
        Eigen::VectorXf U0;
        Eigen::VectorXf UTemp;
        
        Eigen::VectorXf S;
        Eigen::VectorXf S0;
        Eigen::VectorXf STemp;

        Eigen::SparseVector<float> F;
        Eigen::VectorXf FTemp;
        Eigen::SparseVector<float> SF;
        Eigen::VectorXf SFTemp;

        fluidBase(){}

        virtual void interpolateForce(const std::vector<Force> forces){}
        
        virtual void interpolateSource( const std::vector<Source> sources){}

        ~fluidBase(){}
};
} // namespace jfs

#endif