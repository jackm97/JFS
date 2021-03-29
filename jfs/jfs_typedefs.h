#ifndef JFS_TYPEDEFS_H
#define JFS_TYPEDEFS_H

namespace jfs {

typedef enum {
    ZERO,
    PERIODIC,
    DAMPED
} BoundType;

typedef enum {
    VECTOR_FIELD,
    SCALAR_FIELD
} FieldType;

struct Force {
    float pos[3]{0.f};
    float force[3]{0.f};
};

struct Source {
    float pos[3]{0.f};
    float color[3]{0.f};

    float strength = 0.f;
};

}

#endif 