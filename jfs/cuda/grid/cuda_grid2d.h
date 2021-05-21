#ifndef CUDA_GRID2D_H
#define CUDA_GRID2D_H

#include "../cuda_macros.h"


namespace jfs {

    typedef unsigned int uint;

    namespace FieldType2D {
        enum : uint {
            Scalar = 1,
            Vector = 2
        };
    };

    enum : uint {
        CudaGridAsync,
        CudaGridSync
    };

    /// grid can be used in host or device code but cannot be passed between host and device
    ///
    /// \tparam Options - FieldType2D::Scalar or FieldType2D::Vector
    template<uint Options>
    class CudaGrid2D {
    public:

        /// __host__ __device__ empty constructor
        __HOST__DEVICE__
        CudaGrid2D() {}

        /// __host__ __device__ CUDA grid constructor
        ///
        /// \param size - grid size
        /// \param fields - number of fields (i.e. RGB color grid would have 3 fields)
        __HOST__DEVICE__
        CudaGrid2D(uint size, uint fields);

        __HOST__DEVICE__
        /// __host__ __device__ copy constructor
        ///
        /// \param src - grid to be copied
        CudaGrid2D(const CudaGrid2D<Options> &src);

        __HOST__DEVICE__
        /// __host__ __device__ grid constructor. Copies data to device.
        ///
        /// \param data - If called on host, data is on host. If called on device, data is on device.
        /// \param size - grid size
        /// \param fields - number of fields (i.e. RGB color grid would have 3 fields)
        CudaGrid2D(const float *data, uint size, uint fields, uint mode = CudaGridSync);

        /// __host__ __device__ resize grid
        ///
        /// \param size - grid size
        /// \param fields - number of fields (i.e. RGB color grid would have 3 fields)
        __HOST__DEVICE__
        void Resize(uint size, uint fields);

        /// __host__ copy data that is stored on device to grid
        ///
        /// \param data - pointer to device data
        /// \param size - grid size
        /// \param fields - number of fields (i.e. RGB color grid would have 3 fields)
        __HOST__
        void CopyDeviceData(const float *data, uint size, uint fields, uint mode = CudaGridSync);

        /// __host__ __device__ map device data to grid, mapped data is never freed
        ///
        /// \param data - pointer to device data
        /// \param size - grid size
        /// \param fields - number of fields (i.e. RGB color grid would have 3 fields)
        __HOST__DEVICE__
        void MapData(float *data, uint size, uint fields);

        /// __host__ set entire field and dimension of grid to value
        ///
        /// \param val - quantity to set grid to
        /// \param f - field index (i.e. 1 for green in RGB scalar grid)
        /// \param d - field dimension (i.e. 1 for y dimension in vector grid)
        __HOST__
        void SetGridToValue(float val, uint f, uint d, uint mode = CudaGridSync);

        /// __host__ __device__
        ///
        /// \return number of grid nodes per side
        __HOST__DEVICE__
        uint Size() { return size_; }

        /// __host__ __device__
        ///
        /// \return number of fields stored in grid (i.e. RGB scalar grid would return 3)
        __HOST__DEVICE__
        uint Fields() { return fields_; }

        /// __host__ syncs device data with host data
        __HOST__
        void SyncDeviceWithHost(uint mode = CudaGridSync);

        /// __host__ __device__
        /// \note The device data is not automatically synced with the host data. Make sure to call CudaGrid2D::SyncDeviceWithHost
        /// if necessary
        ///
        /// \return pointer to device data
        __HOST__DEVICE__
        float *Data() { return data_; };

        /// __host__ syncs host data with device data
        __HOST__
        void SyncHostWithDevice();

        /// __host__
        /// \note The host data is not automatically synced with the device data. Make sure to call CudaGrid2D::SyncHostWithDevice
        ///
        /// \return pointer to host data
        __HOST__
        float *HostData();

        /// __host__ insert quantity into grid
        /// \note There are no protections against indexing out of bounds of array. It may or may not
        /// result in a segfault; behavior is undefined.
        ///
        /// \param val - inserted value
        /// \param i - x index
        /// \param j - y index
        /// \param f - field index (i.e. 1 for green in RGB scalar grid)
        /// \param d - field dimension (i.e. 1 for y dimension in vector grid)
        __HOST__
        void Insert(float val, uint i, uint j, uint f, uint d, uint mode = CudaGridSync);

        __HOST__
        float Index(uint i, uint j, uint f, uint d);

        /// __host__ __device__ linearly interpolate quantity to grid
        ///
        /// \param q - interpolated quantity
        /// \param i - x position
        /// \param j - y position
        /// \param f - field index (i.e. 1 for green in RGB scalar grid)
        /// \param d - field dimension (i.e. 1 for y dimension in vector grid)
        __HOST__DEVICE__
        void InterpToGrid(float q, float i, float j, uint f, uint d, uint mode = CudaGridSync);

        /// __host__ __device__ interpolate quantity from grid
        /// \note Make sure to call CudaGrid2D::SyncHostWithDevice before calling on the host.
        ///
        /// \param i - x position
        /// \param j - y position
        /// \param f - field index (i.e. 1 for green in RGB scalar grid)
        /// \param d - field dimension (i.e. 1 for y dimension in vector grid)
        ///
        /// \return interpolated quantity
        __HOST__DEVICE__
        float InterpFromGrid(float i, float j, uint f, uint d);

        /// __host__ __device__ overloaded assignment
        ///
        /// \param src - const reference to source grid
        __HOST__DEVICE__
        CudaGrid2D<Options> &operator=(const CudaGrid2D<Options> &src);

        /// __host__ __device__ index grid
        /// \note This indexing allows for easy modification of the grid on the host or the device, however, once all
        /// modifications have been made on the host, a call to CudaGrid2D::SyncDeviceWithHost needs to be made. Along
        /// the same lines, make sure to call CudaGrid2D::SyncHostWithDevice *before* indexing the grid on the host.
        ///
        /// \param i - x index
        /// \param j - y index
        /// \param f - field index (i.e. 1 for green in RGB scalar grid)
        /// \param d - field dimension (i.e. 1 for y dimension in vector grid)
        ///
        /// \return reference to indexed quantity in grid (reference is from host data if called from host and device
        /// data if called from device)
        __HOST__DEVICE__
        float &operator()(int i, int j, int f, int d);

        __HOST__DEVICE__
        ~CudaGrid2D() { FreeGridData(); }

    private:

        // allocation flag
        bool is_allocated_ = false;

        // grid size
        uint size_;
        // number of fields per node
        uint fields_ = 1;
        // number of dims per node
        const uint dims_ = Options;

        // cuda device data
        float *data_;

        // true if using mapped device data
        bool mapped_data_ = false;

        // host specific members
#if !defined(__CUDA_ARCH__) || defined(__PARSE_HOST__)
        // host data
        float* host_data_;
#endif

    private:
        __HOST__DEVICE__
        void FreeGridData();
    };

    template
    class CudaGrid2D<FieldType2D::Scalar>;

    template
    class CudaGrid2D<FieldType2D::Vector>;

} // namespace jfs

#endif