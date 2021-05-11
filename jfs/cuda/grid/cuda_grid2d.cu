#include "./cuda_grid2d.h"
#include <cuda_runtime_api.h>

namespace jfs {

    template<uint Options>
    __HOST__DEVICE__
    CudaGrid2D<Options>::CudaGrid2D(uint size, uint fields) {
        Resize(size, fields);
    }

    template<uint Options>
    __HOST__DEVICE__
    CudaGrid2D<Options>::CudaGrid2D(const CudaGrid2D<Options> &src) {
        *this = src;
    }

    template<uint Options>
    __HOST__DEVICE__
    CudaGrid2D<Options>::CudaGrid2D(const float *data, uint size, uint fields) {
        Resize(size, fields);
        int total_size = size_ * size_ * dims_ * fields_;

#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        memcpy(host_data_, data, total_size*sizeof(float));
        cudaMemcpy(data_, data, total_size*sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(data_, data, total_size * sizeof(float));
#endif
    }

    template<uint Options>
    __HOST__DEVICE__
    void CudaGrid2D<Options>::Resize(uint size, uint fields) {
        FreeGridData();

        size_ = size;
        if (fields != 0)
            fields_ = fields;
        int total_size = size_ * size_ * dims_ * fields_;

#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        host_data_ = (float*) malloc(total_size*sizeof(float));
#endif
        cudaMalloc(&data_, total_size * sizeof(float));

        is_allocated_ = true;
    }

    template<uint Options>
    __HOST__
    void CudaGrid2D<Options>::CopyDeviceData(const float *data, uint size, uint fields) {
        Resize(size, fields);
        int total_size = size_ * size_ * dims_ * fields_;

#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        cudaMemcpy(host_data_, data, total_size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(data_, data, total_size*sizeof(float), cudaMemcpyDeviceToDevice);
#else
        memcpy(data_, data, total_size * sizeof(float));
#endif

        mapped_data_ = false;
    }

    template<uint Options>
    __HOST__DEVICE__
    void CudaGrid2D<Options>::MapData(float *data, uint size, uint fields) {
        FreeGridData();

        size_ = size;
        fields_ = fields;

        data_ = data;

#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        int total_size = (int)size_*(int)size_*(int)dims_*(int)fields_;
        host_data_ = (float*) malloc(total_size*sizeof(float));
        cudaMemcpy(host_data_, data_, total_size*sizeof(float), cudaMemcpyDeviceToHost);
#endif

        is_allocated_ = true;
        mapped_data_ = true;
    }

    /*
    *
    CUDA KERNEL
    *
    */
    template<uint Options>
    __global__
    void setGridKernel(float val, uint f, uint d, float *grid_data, uint grid_size, uint fields) {
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        uint j = blockIdx.y * blockDim.y + threadIdx.y;

#if defined(__CUDA_ARCH__) && !defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        if (i >= grid_size || j >= grid_size)
            return;
        CudaGrid2D<Options> grid;
        grid.MapData(grid_data, grid_size, fields);
        grid(i, j, f, d) = val;
#endif
    }

    /*
    *
    END CUDA KERNEL
    *
    */

    template<uint Options>
    __HOST__
    void CudaGrid2D<Options>::SetGridToValue(float val, uint field, uint dim) {
        dim3 threads_per_block(16, 16);
        dim3 num_blocks(size_ / threads_per_block.x + 1, size_ / threads_per_block.y + 1);

        setGridKernel < Options > <<<num_blocks, threads_per_block>>>(val, field, dim, data_, size_, fields_);

        cudaDeviceSynchronize();
    }

    template<uint Options>
    __HOST__
    float *CudaGrid2D<Options>::HostData() {
#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        return host_data_;
#else
        return nullptr; // this is just here to stop compiler warning, not a device function
#endif
    }

    /*
    *
    CUDA KERNEL
    *
    */
    template<uint Options>
    __global__
    void
    InterpToGridKernel(float val, float i, float j, uint f, uint d, float *grid_data, uint grid_size, uint fields) {
#if defined(__CUDA_ARCH__) && !defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        CudaGrid2D<Options> grid;
        grid.MapData(grid_data, grid_size, fields);
        grid.InterpToGrid(val, i, j, f, d);
#endif
    }

    /*
    *
    END CUDA KERNEL
    *
    */

    template<uint Options>
    __HOST__DEVICE__
    void CudaGrid2D<Options>::InterpToGrid(float q, float i, float j, uint f, uint d) {
#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE

        InterpToGridKernel<Options> <<<1, 1>>>(q, i, j, f, d, data_, size_, fields_);

        cudaDeviceSynchronize();

#else

        int i0 = (int) i;
        int j0 = (int) j;
        CudaGrid2D<Options> &grid = *this;

        for (int di = 0; di < 2; di++) {
            for (int dj = 0; dj < 2; dj++) {
                int i_tmp = i0 + di;
                int j_tmp = j0 + dj;

                float part = abs(((float) i_tmp - i) * ((float) j_tmp - j));
                i_tmp = (i_tmp == i0) ? (i0 + 1) : i0;
                j_tmp = (j_tmp == j0) ? (j0 + 1) : j0;

                if (i_tmp == size_ || j_tmp == size_)
                    continue;
                grid(i_tmp, j_tmp, f, d) += part * q;
            }
        }

#endif
    }

    /*
    *
    CUDA KERNEL
    *
    */
    template<uint Options>
    __global__
    void
    InterpFromGridKernel(float* q_ptr, float i, float j, uint f, uint d, float *grid_data, uint grid_size, uint fields) {
#if defined(__CUDA_ARCH__) && !defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        CudaGrid2D<Options> grid;
        grid.MapData(grid_data, grid_size, fields);
        *q_ptr = grid.InterpFromGrid(i, j, f, d);
#endif
    }

    /*
    *
    END CUDA KERNEL
    *
    */

    template<uint Options>
    __HOST__DEVICE__
    float CudaGrid2D<Options>::InterpFromGrid(float i, float j, uint f, uint d) {
#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        float q;
        float* q_device_ptr;
        cudaMalloc(&q_device_ptr, sizeof(float));

        InterpFromGridKernel<Options> <<<1, 1>>>(q_device_ptr, i, j, f, d, data_, size_, fields_);
        cudaDeviceSynchronize();

        cudaMemcpy(&q, q_device_ptr, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(q_device_ptr);

        return q;
#else

        int i0 = (int) i;
        int j0 = (int) j;
        CudaGrid2D<Options> &grid = *this;

        float q = 0;

        for (int di = 0; di < 2; di++) {
            for (int dj = 0; dj < 2; dj++) {
                int i_tmp = i0 + di;
                int j_tmp = j0 + dj;

                float part = abs(((float) i_tmp - i) * ((float) j_tmp - j));
                i_tmp = (i_tmp == i0) ? (i0 + 1) : i0;
                j_tmp = (j_tmp == j0) ? (j0 + 1) : j0;

                if (i_tmp == size_ || j_tmp == size_)
                    continue;
                q += part * grid(i_tmp, j_tmp, f, d);
            }
        }

        return q;

#endif
    }

    template<uint Options>
    __HOST__DEVICE__
    CudaGrid2D<Options> &CudaGrid2D<Options>::operator=(const CudaGrid2D<Options> &src) { // NOLINT(bugprone-unhandled-self-assignment)
        Resize(src.size_, src.fields_);
        int total_size = size_ * size_ * dims_ * fields_;

#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        memcpy(host_data_, src.host_data_, total_size*sizeof(float));
        cudaMemcpy(data_, src.data_, total_size*sizeof(float), cudaMemcpyDeviceToDevice);
#else
        memcpy(data_, src.data_, total_size * sizeof(float));
#endif

        mapped_data_ = false;

        return *this;
    }

    template<uint Options>
    __HOST__DEVICE__
    float &CudaGrid2D<Options>::operator()(int i, int j, int f, int d)
    {
        int offset = size_ * fields_ * dims_ * j + fields_ * dims_ * i +
                     dims_ * f + d;

#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        return *(host_data_ + offset);
#else
        return *(data_ + offset);
#endif
    }

    template<uint Options>
    __HOST__DEVICE__
    void CudaGrid2D<Options>::FreeGridData() {
        if (!is_allocated_)
            return;

#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        free(host_data_);
#endif
        if (!mapped_data_)
            cudaFree(data_);

        is_allocated_ = false;
    }

    template<uint Options>
    __HOST__
    void CudaGrid2D<Options>::Insert(float val, uint i, uint j, uint f, uint d) {
        int offset = size_ * fields_ * dims_ * j + fields_ * dims_ * i +
                     dims_ * f + d;
        cudaMemcpy(data_ + offset, &val, sizeof(float), cudaMemcpyHostToDevice);
    }

    template<uint Options>
    __HOST__
    void CudaGrid2D<Options>::SyncDeviceWithHost() {

#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        cudaMemcpy(data_, host_data_, size_*size_*fields_*dims_*sizeof(float), cudaMemcpyHostToDevice);
#endif
    }

    template<uint Options>
    __HOST__
    void CudaGrid2D<Options>::SyncHostWithDevice() {

#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__) // __PARSE_HOST__ is used to toggle parse of host code in IDE
        cudaMemcpy(host_data_, data_, size_*size_*fields_*dims_*sizeof(float), cudaMemcpyDeviceToHost);
#endif
    }

    template __global__ void
    InterpToGridKernel<FieldType2D::Scalar>(float val, float i, float j, uint f, uint d, float *grid_data,
                                            uint grid_size, uint fields);

    template __global__ void
    InterpToGridKernel<FieldType2D::Vector>(float val, float i, float j, uint f, uint d, float *grid_data,
                                            uint grid_size, uint fields);

    template __global__
    void setGridKernel<FieldType2D::Scalar>(float val, uint f, uint d, float *grid_data, uint grid_size, uint fields);

    template __global__
    void setGridKernel<FieldType2D::Vector>(float val, uint f, uint d, float *grid_data, uint grid_size, uint fields);

} // namespace jfs

