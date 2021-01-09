#ifndef FLUIDBASE_H
#define FLUIDBASE_H
#include "../jfs_inline.h"

namespace jfs {

class fluidBase {
    public:
        fluidBase(){}
         virtual void resetFluid(){}

        virtual void getImage(Eigen::VectorXf img){}

        // returns true if the step failed
        virtual bool calcNextStep(const std::vector<Force> forces, const std::vector<Source> sources){return false;}

        ~fluidBase(){}

    protected:
        Eigen::VectorXf U; 
        Eigen::VectorXf U0;
        
        Eigen::VectorXf S;
        Eigen::VectorXf S0;

        Eigen::SparseVector<float> F;
        Eigen::SparseVector<float> SF;

        SparseMatrix LAPLACEX; // scalar laplace extended for x concatenated fields

        // returns true if the step failed
        virtual bool calcNextStep( ){return false;}
        
        virtual void initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dt){}

        virtual void interpolateForce(const std::vector<Force> forces){}
        
        virtual void interpolateSource( const std::vector<Source> sources){}
};
} // namespace jfs

#endif