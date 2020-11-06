#ifndef BOUNDTYPES_H
#define BOUNDTYPES_H

#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace jfs {

typedef enum {
    ZERO,
    PERIODIC
} BOUND_TYPE;

typedef Eigen::Vector3f ColorRGB;

}

#endif 