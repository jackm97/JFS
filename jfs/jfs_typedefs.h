#ifndef BOUNDTYPES_H
#define BOUNDTYPES_H

#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace jfs {

typedef enum {
    ZERO,
    PERIODIC
} BOUND_TYPE;

typedef enum {
    VECTOR_FIELD,
    SCALAR_FIELD
} FIELD_TYPE;

typedef Eigen::Vector3f ColorRGB;
typedef Eigen::SparseMatrix<float> SparseMatrix;

struct Force {
    float x=0;
    float y=0;
    float z=0;

    float Fx=0;
    float Fy=0;
};

struct Source {
    float x=0;
    float y=0;
    float z=0;

    ColorRGB color={0,0,0};

    float strength;
};

}

#endif 