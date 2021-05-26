cmake_minimum_required(VERSION 3.17)
project(jfs VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

#####################################################################
# CUSTOM OPTIONS:
#
# EIGEN PATH:
#   If adding this as a CMake subdirectory
#   EIGEN_PATH can instead be set in the top
#   level CMakeLists.txt
    if(NOT DEFINED EIGEN_PATH)    
        set(EIGEN_PATH ${PROJECT_SOURCE_DIR}/extern/eigen)
    endif()

# END CUSTOM OPTIONS
#####################################################################

include_directories(${PROJECT_SOURCE_DIR}
                    ${EIGEN_PATH})

set(JFS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/jfs)


# Source Files
set(JFS_SRC
        ${JFS_SOURCE_DIR}/grid/grid2D.cpp
        ${JFS_SOURCE_DIR}/grid/grid3D.cpp
        ${JFS_SOURCE_DIR}/differential_ops/grid_diff2d.cpp
        ${JFS_SOURCE_DIR}/differential_ops/grid_diff3d.cpp
        ${JFS_SOURCE_DIR}/visualization/grid_smoke2d.cpp
        ${JFS_SOURCE_DIR}/visualization/grid_smoke3d.cpp
        ${JFS_SOURCE_DIR}/JSSFSolver.cpp
        ${JFS_SOURCE_DIR}/JSSFSolver3D.cpp
        ${JFS_SOURCE_DIR}/LBMSolver.cpp)
set(JFS_HEADERS
        ${JFS_SOURCE_DIR}/grid/grid2D.h
        ${JFS_SOURCE_DIR}/grid/grid3D.h
        ${JFS_SOURCE_DIR}/differential_ops/grid_diff2d.h
        ${JFS_SOURCE_DIR}/differential_ops/grid_diff3d.h
        ${JFS_SOURCE_DIR}/visualization/grid_smoke2d.h
        ${JFS_SOURCE_DIR}/visualization/grid_smoke3d.h
        ${JFS_SOURCE_DIR}/JSSFSolver.h
        ${JFS_SOURCE_DIR}/JSSFSolver3D.h
        ${JFS_SOURCE_DIR}/LBMSolver.h)

# CUDA Files
set(JFS_CUDA_SRC
        ${JFS_SOURCE_DIR}/cuda/grid/cuda_grid2d.cu
        ${JFS_SOURCE_DIR}/cuda/lbm_solver_cuda.cu)
set(JFS_CUDA_HEADERS
        ${JFS_SOURCE_DIR}/cuda/grid/cuda_grid2d.h
        ${JFS_SOURCE_DIR}/cuda/lbm_solver_cuda.h)



# Add Library
add_definitions(-DJFS_STATIC)
add_library(jfs STATIC ${JFS_SRC} ${JFS_HEADERS} ${JFS_CUDA_SRC} ${JFS_CUDA_HEADERS})
set_target_properties(
        jfs
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF)
set_property(TARGET jfs PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET jfs PROPERTY CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)
find_library(CUDADEVRT cudadevrt "${CUDAToolkit_LIBRARY_DIR}")

target_link_libraries(jfs PRIVATE CUDA::cudart_static CUDADEVRT)
    