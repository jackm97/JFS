#include <jfs/base/grid2D.h>

namespace jfs {

JFS_INLINE void grid2D::initializeGrid(unsigned int N, float L, BOUND_TYPE BOUND, float dt)
{

    initializeGridProperties(N, L, BOUND, dt);

    Laplace(LAPLACE, 1);
    Laplace(VEC_LAPLACE, 2);
    div(DIV);
    grad(GRAD);
}


JFS_INLINE void grid2D::satisfyBC(Eigen::VectorXf &u)
{
    int i,j;
    if (BOUND == PERIODIC)
    for (int idx=0; idx < N; idx++)
    {
        // top
        i = idx;
        j = N-1;
        u(N*N*0 + N*j + i) = u(N*N*0 + N*(j-(N-1)) + i);
        u(N*N*1 + N*j + i) = u(N*N*1 + N*(j-(N-1)) + i);

        // bottom
        i = idx;
        j = 0;
        u(N*N*0 + N*j + i) = u(N*N*0 + N*(j+(N-1)) + i);
        u(N*N*1 + N*j + i) = u(N*N*1 + N*(j+(N-1)) + i);

        // left
        j = idx;
        i = N-1;
        u(N*N*0 + N*j + i) = u(N*N*0 + N*j + (i-(N-1)));
        u(N*N*1 + N*j + i) = u(N*N*1 + N*j + (i-(N-1)));

        // right
        j = idx;
        i = 0;
        u(N*N*0 + N*j + i) = u(N*N*0 + N*j + (i+(N-1)));
        u(N*N*1 + N*j + i) = u(N*N*1 + N*j + (i+(N-1)));
    }

    else if (BOUND == ZERO)
    for (int idx=0; idx < N; idx++)
    {
        // top
        i = idx;
        j = N-1;
        u(N*N*1 + N*j + i) = 0;

        // bottom
        i = idx;
        j = 0;
        u(N*N*1 + N*j + i) = 0;

        // left
        j = idx;
        i = N-1;
        u(N*N*0 + N*j + i) = 0;

        // right
        j = idx;
        i = 0;
        u(N*N*0 + N*j + i) = 0;
    }
}


JFS_INLINE void grid2D::Laplace(SparseMatrix &dst, unsigned int dims, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*5*dims);

    for (int field = 0; field < fields; field++)
    {
        for (int dim = 0; dim < dims; dim++)
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    int iMat = field*dims*N*N + dim*N*N + j*N + i;
                    int jMat = iMat;
                    
                    // x,y
                    tripletList.push_back(T(iMat,jMat,-4.f));
                    
                    //x,y+h
                    jMat = field*dims*N*N + dim*N*N + (j+1)*N + i;
                    if ((j+1) >= N)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N + dim*N*N + 1*N + i;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x,y-h
                    jMat = field*dims*N*N + dim*N*N + (j-1)*N + i;
                    if ((j-1) < 0)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N + dim*N*N + (N-2)*N + i;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x+h,y
                    jMat = field*dims*N*N + dim*N*N + j*N + (i+1);
                    if ((i+1) >= N)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N + dim*N*N + j*N + 1;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                    
                    //x-h,y
                    jMat = field*dims*N*N + dim*N*N + j*N + i-1;
                    if ((i-1) < 0)
                        switch (BOUND)
                        {
                            case ZERO:
                                break;

                            case PERIODIC:
                                jMat = field*dims*N*N + dim*N*N + j*N + N-2;  
                                tripletList.push_back(T(iMat,jMat,1.f)); 
                        }
                    else
                        tripletList.push_back(T(iMat,jMat,1.f));
                }
            }
        }
    }
    dst = SparseMatrix(N*N*dims*fields,N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(D*D) * dst;
}


JFS_INLINE void grid2D::div(SparseMatrix &dst, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*2*2);

    int dims = 2;

    for (int field=0; field < fields; field++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int iMat = j*N + i;
                int jMat = iMat;

                int dim = 0;
                
                //x+h,y
                jMat = field*dims*N*N + dim*N*N + j*N + (i+1);
                if ((i+1) >= N)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + dim*N*N + j*N + 1;  
                            tripletList.push_back(T(iMat,jMat,1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,1.f));
                
                //x-h,y
                jMat = field*dims*N*N + dim*N*N + j*N + i-1;
                if ((i-1) < 0)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + dim*N*N + j*N + N-2;  
                            tripletList.push_back(T(iMat,jMat,-1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,-1.f));

                dim = 1;
                
                //x,y+h
                jMat = field*dims*N*N + dim*N*N + (j+1)*N + i;
                if ((j+1) >= N)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + dim*N*N + 1*N + i;  
                            tripletList.push_back(T(iMat,jMat,1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,1.f));
                
                //x,y-h
                jMat = dim*N*N + (j-1)*N + i;
                if ((j-1) < 0)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = dim*N*N + (N-2)*N + i;  
                            tripletList.push_back(T(iMat,jMat,-1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,-1.f));
            }
        }
    }
    dst = SparseMatrix(N*N*fields,N*N*2*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}


JFS_INLINE void grid2D::grad(SparseMatrix &dst, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*2*2);

    int dims = 1;

    for (int field=0; field < fields; field++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int iMat = j*N + i;
                int jMat = iMat;

                int dim = 0;
                iMat = field*2*N*N + dim*N*N + j*N + i;
                
                //x+h,y
                jMat = field*dims*N*N + j*N + (i+1);
                if ((i+1) >= N)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + j*N + 1;  
                            tripletList.push_back(T(iMat,jMat,1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,1.f));
                
                //x-h,y
                jMat = field*dims*N*N + j*N + i-1;
                if ((i-1) < 0)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + j*N + N-2;  
                            tripletList.push_back(T(iMat,jMat,-1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,-1.f));

                dim = 1;
                iMat = field*2*N*N + dim*N*N + j*N + i;
                
                //x,y+h
                jMat = field*dims*N*N + (j+1)*N + i;
                if ((j+1) >= N)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + 1*N + i;  
                            tripletList.push_back(T(iMat,jMat,1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,1.f));
                
                //x,y-h
                jMat = field*dims*N*N + (j-1)*N + i;
                if ((j-1) < 0)
                    switch (BOUND)
                    {
                        case ZERO:
                            break;

                        case PERIODIC:
                            jMat = field*dims*N*N + (N-2)*N + i;  
                            tripletList.push_back(T(iMat,jMat,-1.f)); 
                    }
                else
                    tripletList.push_back(T(iMat,jMat,-1.f));
            }
        }
    }

    dst = SparseMatrix(N*N*2*fields,N*N*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
    dst = 1.f/(2*D) * dst;
}

JFS_INLINE void grid2D::backstream(Eigen::VectorXf &dst, const Eigen::VectorXf &src, const Eigen::VectorXf &ufield, float dt, int dims, int fields)
{
    for (int index = 0; index < N*N; index++)
    {
        int j = std::floor(index/N);
        int i = index - N*j;
        Eigen::VectorXf X(2);
        X(0) = D*(i + .5);
        X(1) = D*(j + .5);

        X = sourceTrace(X, ufield, 2, -dt);

        Eigen::VectorXf interp_indices = (X.array())/D - .5;

        Eigen::VectorXf interp_quant = calcLinInterp(interp_indices, src, dims, fields);

        Eigen::VectorXi insert_indices(2);
        insert_indices(0) = i;
        insert_indices(1) = j;

        insertIntoField(insert_indices, interp_quant, dst, dims, fields);
    }
}

JFS_INLINE Eigen::VectorXf grid2D::indexField(Eigen::VectorXi indices, const Eigen::VectorXf &src, int dims, int fields)
{
    int i = indices(0);
    int j = indices(1);

    Eigen::VectorXf indexed_quant(dims*fields);

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            indexed_quant(dims*f + d) = src(N*N*dims*f + N*N*d + N*j + i);
        }
    }

    return indexed_quant;
}

JFS_INLINE void grid2D::insertIntoField(Eigen::VectorXi indices, Eigen::VectorXf q, Eigen::VectorXf &dst, int dims, int fields)
{
    int i = indices(0);
    int j = indices(1);

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            dst(N*N*dims*f + N*N*d + N*j + i) = q(dims*f + d);
        }
    }
}


JFS_INLINE Eigen::VectorXf grid2D::calcLinInterp(Eigen::VectorXf interp_indices, const Eigen::VectorXf &src, int dims, unsigned int fields)
{
    Eigen::VectorXf interp_quant(dims*fields);

    float i0 = interp_indices(0);
    float j0 = interp_indices(1);

    switch (BOUND)
    {
        case ZERO:
            i0 = (i0 < 0) ? 0:i0;
            i0 = (i0 > (N-1)) ? (N-1):i0;
            j0 = (j0 < 0) ? 0:j0;
            j0 = (j0 > (N-1)) ? (N-1):j0;
            break;
        
        case PERIODIC:
            while (i0 < 0 || i0 > N-1 || j0 < 0 || j0 > N-1)
            {
                i0 = (i0 < 0) ? (N-1+i0):i0;
                i0 = (i0 > (N-1)) ? (i0 - (N-1)):i0;
                j0 = (j0 < 0) ? (N-1+j0):j0;
                j0 = (j0 > (N-1)) ? (j0 - (N-1)):j0;
            }
            break;
    }

    int i0_floor = (int) i0;
    int j0_floor = (int) j0;
    int i0_ceil = i0_floor + 1;
    int j0_ceil = j0_floor + 1;
    int iMat, jMat, iref, jref;
    float part;

    for (int f = 0; f < fields; f++)
    {
        for (int d = 0; d < dims; d++)
        {
            part = (i0_ceil - i0)*(j0_ceil - j0);
            part = std::abs(part);
            float q1 = part * src(N*N*dims*f + N*N*d + N*j0_floor + i0_floor);

            float q2;
            if (j0_ceil < N)
            {
                part = (i0_ceil - i0)*(j0_floor - j0);
                part = std::abs(part);
                q2 = part * src(N*N*dims*f + N*N*d + N*j0_ceil + i0_floor); 
            }
            else
            {
                q2 = 0;
            }

            float q3;
            if (i0_ceil < N)
            {
                part = (i0_floor - i0)*(j0_ceil - j0);
                part = std::abs(part);
                q3 = part * src(N*N*dims*f + N*N*d + N*j0_floor + i0_ceil); 
            }
            else
            {
                q3 = 0;
            }

            float q4;
            if (i0_ceil < N && j0_ceil < N)
            {
                part = (i0_floor - i0)*(j0_floor - j0);
                part = std::abs(part);
                q4 = part * src(N*N*dims*f + N*N*d + N*j0_ceil + i0_ceil); 
            }
            else
            {
                q4 = 0;
            }

            interp_quant(dims*f + d) = q1 + q2 + q3 + q4;
        }
    }

    return interp_quant;
}

} // namespace jfs

