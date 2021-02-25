#ifndef JFS_TYPEDEFS_H
#define JFS_TYPEDEFS_H

#include <Eigen/Eigen>

namespace jfs {

typedef enum {
    ZERO,
    PERIODIC
} BOUND_TYPE;

typedef enum {
    VECTOR_FIELD,
    SCALAR_FIELD
} FIELD_TYPE;

struct Force {
    float x=0;
    float y=0;
    float z=0;

    float Fx=0;
    float Fy=0;
    float Fz=0;
};

struct Source {
    float x=0;
    float y=0;
    float z=0;

    Eigen::Vector3f color={0,0,0};

    float strength;
};

}

#endif 