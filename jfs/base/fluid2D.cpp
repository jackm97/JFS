#include <jfs/base/fluid2D.h>

namespace jfs {


JFS_INLINE void fluid2D::initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dt)
{

    initializeGrid(N, L, BOUND, dt);

    U.resize(N*N*2);

    F.resize(N*N*2);
    
    S.resize(3*N*N);
    SF.resize(3*N*N);
    
    Laplace(LAPLACEX, 1, 3); // sources have three color channels

    resetFluid();
}


JFS_INLINE void fluid2D::resetFluid()
{
    U.setZero();
    U0 = U;

    F.setZero();

    S.setZero();
    S0 = S;
    SF.setZero();
}

JFS_INLINE void fluid2D::getImage(Eigen::VectorXf &image)
{
    if (image.rows() != N*N*3)
        image.resize(N*N*3);

    for (int i=0; i < N; i++)
        for (int j=0; j < N; j++)
        {
            image(N*3*j + 0 + i*3) = S(0*N*N + N*j + i);
            image(N*3*j + 1 + i*3) = S(1*N*N + N*j + i);
            image(N*3*j + 2 + i*3) = S(2*N*N + N*j + i);
        }
    image = (image.array() <= 1.).select(image, 1.);
}


JFS_INLINE void fluid2D::interpolateForce(const std::vector<Force> forces)
{
    for (int f=0; f < forces.size(); f++)
    {
        const Force &force = forces[f];
        if (force.x>L || force.y>L) continue;
        
        float i = force.x/D;
        float j = force.y/D;

        int i0 = std::floor(i);
        int j0 = std::floor(j);

        int dims = 2;
        float fArr[2] = {force.Fx, force.Fy};
        for (int dim=0; dim < dims; dim++)
        {
            F.insert(N*N*dim + N*j0 + i0) += fArr[dim]*std::abs((j0+1 - j)*(i0+1 - i));
            if (i0 < (N-1))
                F.insert(N*N*dim + N*j0 + (i0+1)) += fArr[dim]*std::abs((j0+1 - j)*(i0 - i));
            if (j0 < (N-1))
                F.insert(N*N*dim + N*(j0+1) + i0) += fArr[dim]*std::abs((j0 - j)*(i0+1 - i));
            if (i0 < (N-1) && j0 < (N-1))
                F.insert(N*N*dim + N*(j0+1) + (i0+1)) += fArr[dim]*std::abs((j0 - j)*(i0 - i));
        }
    }
}


JFS_INLINE void fluid2D::interpolateSource(const std::vector<Source> sources)
{
    for (int f=0; f < sources.size(); f++)
    {
        const Source &source = sources[f];
        if (source.x>L || source.y>L) continue;
        
        float i = source.x/D;
        float j = source.y/D;

        int i0 = std::floor(i);
        int j0 = std::floor(j);

        for (int c=0; c < 3; c++)
        {
            float cval = {source.color(c) * source.strength};
            SF.insert(c*N*N + N*j0 + i0) += cval*std::abs((j0+1 - j)*(i0+1 - i));
            if (i0 < (N-1))
                SF.insert(c*N*N + N*j0 + (i0+1)) += cval*std::abs((j0+1 - j)*(i0 - i));
            if (j0 < (N-1))
                SF.insert(c*N*N + N*(j0+1) + i0) += cval*std::abs((j0 - j)*(i0+1 - i));
            if (i0 < (N-1) && j0 < (N-1))            
                SF.insert(c*N*N + N*(j0+1) + (i0+1)) += cval*std::abs((j0 - j)*(i0 - i));
        }
    }
}

} // namespace jfs