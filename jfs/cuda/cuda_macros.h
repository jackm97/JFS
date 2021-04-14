#ifndef JFS_CUDA_MACROS_H
#define  JFS_CUDA_MACROS_H

// below macros allow header to be included into cpp files compiled without nvcc
#ifndef __CUDA_ARCH__
#define __HOST__DEVICE__
#define __HOST__
#define __DEVICE__
#else
#define __HOST__DEVICE__ __host__ __device__
#define __HOST__ __host__
#define __DEVICE__ __device__
#endif

// enables parsing of host code and disables parsing of device code
// when using CLion
// COMMENT OUT WHEN COMPILING
//#define __PARSE_HOST__

#if defined(__PARSE_HOST__)
#error "#undef __PARSE_HOST__ in cuda_macros.h when compiling"
#endif

#endif