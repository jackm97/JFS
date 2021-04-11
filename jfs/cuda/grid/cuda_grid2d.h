#ifndef CUDA_GRID2D_H
#define CUDA_GRID2D_H

#ifndef __CUDA_ARCH__
    #define __HOST__DEVICE__
    #define __HOST__
    #define __DEVICE__
#else
    #define __HOST__DEVICE__ __host__ __device__
    #define __HOST__ __host__
    #define __DEVICE__ __device__
#endif


namespace jfs {
    
    typedef unsigned int uint;
    
    namespace FieldType2D
    {    
        enum : uint {
            Scalar = 1,
            Vector = 2
        };
    };

    template <uint Options>
    class CudaGrid2D {
        public:
            // constructors
            __HOST__DEVICE__
            CudaGrid2D(){}

            __HOST__DEVICE__
            CudaGrid2D(uint size, uint fields);

            __HOST__DEVICE__
            CudaGrid2D(const CudaGrid2D<Options>& src);

            __HOST__DEVICE__
            // If called on host, data is on host. If called on device, data is on device
            // To copy data from device when called from host, use CopyDeviceData()
            CudaGrid2D(const float* data, uint size, uint fields);
            // end constructors

            // Resize, if fields = 0 the number of fields remains unchaged
            // If grid hasn't been allocated, fields=0 allocates grid with 1 field
            __HOST__DEVICE__
            void Resize(uint size, uint fields = 0);

            // copy data that is stored on device to grid
            __HOST__
            void CopyDeviceData(const float* data, uint size, uint fields);

            // map device data to grid, mapped data is never freed
            __HOST__DEVICE__
            void MapData(float* data, uint size, uint fields);

            // set grid to value
            __HOST__
            void SetGridToValue(float val, uint f, uint d);

            // return grid size
            __HOST__DEVICE__
            uint Size() {return size_;}

            // return number of fields per node
            __HOST__DEVICE__
            uint Fields() {return fields_;}
    
            // returns pointer to device data
            __HOST__DEVICE__
            float* Data() {return data_;};
    
            // returns pointer to host data
            __HOST__
            float* HostData();

            // interpolate quantity to grid
            __HOST__DEVICE__
            void InterpToGrid(float q, float i, float j, uint f, uint d);

            // interpolate quantity from grid
//            __HOST__DEVICE__
//            float InterpFromGrid(float i, float j, uint f, uint d);

            // overloaded operators
            __HOST__DEVICE__
            CudaGrid2D <Options> & operator=(const CudaGrid2D<Options>& src);

            // operator() indexes the grid. When called by device, indexed quantity is returned by reference.
            // When called on host, indexed value is returned by value.
            #ifndef __CUDA_ARCH__
            __HOST__
            float operator()(int i, int j, int f, int d);
            #else
            __DEVICE__
            float& operator()(int i, int j, int f, int d);
            #endif

            __HOST__DEVICE__
            ~CudaGrid2D(){ FreeGridData(); }

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
            float* data_;

            // true if using mapped device data
            bool mapped_data_ = false;

            // host specific members
            #ifndef __CUDA_ARCH__
            // host data
            float* host_data_;
            #endif

        private:
            __HOST__DEVICE__
            void FreeGridData();
    };

} // namespace jfs

#endif