#include <jfs/base/fluid3D.h>

namespace jfs {


JFS_INLINE void fluid3D::initializeFluid(unsigned int N, float L, BOUND_TYPE BOUND, float dt)
{

    initializeGrid(N, L, BOUND, dt);

    U.resize(N*N*N*3);

    F.resize(N*N*N*3);
    
    S.resize(3*N*N*N);
    SF.resize(3*N*N*N);
    
    Laplace(LAPLACEX, 1, 3); // sources have three color channels

    resetFluid();
}


JFS_INLINE void fluid3D::resetFluid()
{
    U.setZero();
    U0 = U;

    F.setZero();

    S.setZero();
    S0 = S;
    SF.setZero();
}

JFS_INLINE void fluid3D::getImage(Eigen::VectorXf &image)
{
    if (image.rows() != N*N*N*3)
        image.resize(N*N*N*3);

    for (int i=0; i < N; i++)
        for (int j=0; j < N; j++)
            for (int k=0; k < N; k++)
        {
            image(N*3*N*k + N*3*j + i*3 + 0) = S(0*N*N*N + N*N*k + N*j + i);
            image(N*3*N*k + N*3*j + i*3 + 1) = S(1*N*N*N + N*N*k + N*j + i);
            image(N*3*N*k + N*3*j + i*3 + 2) = S(2*N*N*N + N*N*k + N*j + i);
        }
    image = (image.array() <= 1.).select(image, 1.);
}


JFS_INLINE void fluid3D::interpolateForce(const std::vector<Force> forces)
{
    for (int f=0; f < forces.size(); f++)
    {
        const Force &force = forces[f];
        if (force.x>L || force.y>L) continue;
        
        float i = force.x/D;
        float j = force.y/D;
        float k = force.z/D;

        int i0 = std::floor(i);
        int j0 = std::floor(j);
        int k0 = std::floor(k);

        int dims = 3;
        float fArr[3] = {force.Fx, force.Fy, force.Fz};
        for (int dim=0; dim < dims; dim++)
        {
            F.insert(N*N*N*dim + N*N*k0 + N*j0 + i0) += fArr[dim]*std::abs((j0+1 - j)*(i0+1 - i)*(k0+1 - i));
            if (i0 < (N-1))
                F.insert(N*N*N*dim + N*N*k0 + N*j0 + (i0+1)) += fArr[dim]*std::abs((j0+1 - j)*(i0 - i)*(k0+1 - i));
            if (j0 < (N-1))
                F.insert(N*N*N*dim + N*N*k0 + N*(j0+1) + i0) += fArr[dim]*std::abs((j0 - j)*(i0+1 - i)*(k0+1 - i));
            if (k0 < (N-1))
                F.insert(N*N*N*dim + N*N*(k0+1) + N*j0 + i0) += fArr[dim]*std::abs((j0+1 - j)*(i0+1 - i)*(k0 - i));
            if (i0 < (N-1) && j0 < (N-1))
                F.insert(N*N*N*dim + N*N*k0 + N*(j0+1) + (i0+1)) += fArr[dim]*std::abs((j0 - j)*(i0 - i)*(k0+1 - i));
            if (i0 < (N-1) && k0 < (N-1))
                F.insert(N*N*N*dim + N*N*(k0+1) + N*j0 + (i0+1)) += fArr[dim]*std::abs((j0+1 - j)*(i0 - i)*(k0 - i));
            if (j0 < (N-1) && k0 < (N-1))
                F.insert(N*N*N*dim + N*N*(k0+1) + N*(j0+1) + i0) += fArr[dim]*std::abs((j0 - j)*(i0+1 - i)*(k0 - i));
            if (i0 < (N-1) && j0 < (N-1) && k0 < (N-1))
                F.insert(N*N*N*dim + N*N*(k0+1) + N*(j0+1) + (i0+1)) += fArr[dim]*std::abs((j0 - j)*(i0 - i)*(k0 - i));
        }
    }
}


JFS_INLINE void fluid3D::interpolateSource(const std::vector<Source> sources)
{
    for (int f=0; f < sources.size(); f++)
    {
        const Source &source = sources[f];
        if (source.x>L || source.y>L) continue;
        
        float i = source.x/D;
        float j = source.y/D;
        float k = source.z/D;

        int i0 = std::floor(i);
        int j0 = std::floor(j);
        int k0 = std::floor(k);

        for (int c=0; c < 3; c++)
        {
            float cval = {source.color(c) * source.strength};

            SF.insert(N*N*N*c + N*N*k0 + N*j0 + i0) += cval*std::abs((j0+1 - j)*(i0+1 - i)*(k0+1 - i));
            if (i0 < (N-1))
                SF.insert(N*N*N*c + N*N*k0 + N*j0 + (i0+1)) += cval*std::abs((j0+1 - j)*(i0 - i)*(k0+1 - i));
            if (j0 < (N-1))
                SF.insert(N*N*N*c + N*N*k0 + N*(j0+1) + i0) += cval*std::abs((j0 - j)*(i0+1 - i)*(k0+1 - i));
            if (k0 < (N-1))
                SF.insert(N*N*N*c + N*N*(k0+1) + N*j0 + i0) += cval*std::abs((j0+1 - j)*(i0+1 - i)*(k0 - i));
            if (i0 < (N-1) && j0 < (N-1))
                SF.insert(N*N*N*c + N*N*k0 + N*(j0+1) + (i0+1)) += cval*std::abs((j0 - j)*(i0 - i)*(k0+1 - i));
            if (i0 < (N-1) && k0 < (N-1))
                SF.insert(N*N*N*c + N*N*(k0+1) + N*j0 + (i0+1)) += cval*std::abs((j0+1 - j)*(i0 - i)*(k0 - i));
            if (j0 < (N-1) && k0 < (N-1))
                SF.insert(N*N*N*c + N*N*(k0+1) + N*(j0+1) + i0) += cval*std::abs((j0 - j)*(i0+1 - i)*(k0 - i));
            if (i0 < (N-1) && j0 < (N-1) && k0 < (N-1))
                SF.insert(N*N*N*c + N*N*(k0+1) + N*(j0+1) + (i0+1)) += cval*std::abs((j0 - j)*(i0 - i)*(k0 - i));
        }
    }
}

} // namespace jfs