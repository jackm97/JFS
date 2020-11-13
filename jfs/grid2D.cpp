#include <jfs/grid2D.h>

namespace jfs {

JFS_INLINE void grid2D::setXGrid()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int dim = 0;
            X(dim*N*N + j*N + i) = D*(i+.5);
            
            dim = 1;
            X(dim*N*N + j*N + i) = D*(j+.5);
        }
    }
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
        u(N*N*1 + N*j + i) = u(N*N*1 + N*(j-(N-1)) + i);

        // bottom
        i = idx;
        j = 0;
        u(N*N*1 + N*j + i) = u(N*N*1 + N*(j+(N-1)) + i);

        // left
        j = idx;
        i = N-1;
        u(N*N*0 + N*j + i) = u(N*N*0 + N*j + (i-(N-1)));

        // right
        j = idx;
        i = 0;
        u(N*N*0 + N*j + i) = u(N*N*0 + N*j + (i+(N-1)));
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
                iMat = field*dims*N*N + dim*N*N + j*N + i;
                
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
                iMat = field*dims*N*N + dim*N*N + j*N + i;
                
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


JFS_INLINE void grid2D::calcLinInterp(SparseMatrix &dst, const Eigen::VectorXf &ij0, int dims, unsigned int fields)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(N*N*4*dims);
    
    for (int field=0; field < fields; field++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float i0 = ij0(0*N*N + N*j + i);
                float j0 = ij0(1*N*N + N*j + i);

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

                for (int dim = 0; dim < dims; dim++)
                {
                    iMat = field*dims*N*N + dim*N*N + j*N + i;
                    
                    jMat = field*dims*N*N + dim*N*N + j0_floor*N + i0_floor;
                    part = (i0_ceil - i0)*(j0_ceil - j0);
                    tripletList.push_back(T(iMat, jMat, std::abs(part)));
                    if (jMat > N*N*dims*fields-1 || jMat < 0)
                    {
                        printf("NO!\n");
                    }
                    
                    jMat = field*dims*N*N + dim*N*N + j0_ceil*N + i0_floor;
                    if (j0_ceil < N)
                    {
                        part = (i0_ceil - i0)*(j0_floor - j0);
                        tripletList.push_back(T(iMat, jMat, std::abs(part)));
                    if (jMat > N*N*dims*fields-1 || jMat < 0)
                    {
                        printf("NO!\n");
                    }
                    }
                    
                    jMat = field*dims*N*N + dim*N*N + j0_floor*N + i0_ceil;
                    if (i0_ceil < N)
                    {
                        part = (i0_floor - i0)*(j0_ceil - j0);
                        tripletList.push_back(T(iMat, jMat, std::abs(part)));
                    if (jMat > N*N*dims*fields-1 || jMat < 0)
                    {
                        printf("NO!\n");
                    }
                    }
                    
                    jMat = field*dims*N*N + dim*N*N + j0_ceil*N + i0_ceil;
                    if (i0_ceil < N && j0_ceil < N)
                    {
                        part = (i0_floor - i0)*(j0_floor - j0);
                        tripletList.push_back(T(iMat, jMat, std::abs(part)));
                    if (jMat > N*N*dims*fields-1 || jMat < 0)
                    {
                        printf("NO!\n");
                    }
                    }
                }
            }
        }        
    }

    dst = SparseMatrix(N*N*dims*fields,N*N*dims*fields);
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
}

} // namespace jfs

