cmake_minimum_required(VERSION 3.19)
project(jfs VERSION 0.1.0 LANGUAGES CUDA)

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
set(JFS_SRC_BASE    ${JFS_SOURCE_DIR}/base/grid2D.cpp
                    ${JFS_SOURCE_DIR}/base/grid3D.cpp)

set(JFS_SRC_DIFF_OPS    ${JFS_SOURCE_DIR}/differential_ops/grid_diff2d.cpp
                        ${JFS_SOURCE_DIR}/differential_ops/grid_diff3d.cpp)

set(JFS_SRC_VIS     ${JFS_SOURCE_DIR}/visualization/grid_smoke2d.cpp
                        ${JFS_SOURCE_DIR}/visualization/grid_smoke3d.cpp)

set(JFS_SRC_TOP ${JFS_SOURCE_DIR}/JSSFSolver.cpp
                ${JFS_SOURCE_DIR}/JSSFSolver3D.cpp
                ${JFS_SOURCE_DIR}/LBMSolver.cpp)

# CUDA Source Files
set(JFS_CUDA_SRC        ${JFS_SOURCE_DIR}/cuda/grid/cuda_grid2d.cu
                        ${JFS_SOURCE_DIR}/cuda/lbm_solver_cuda.cu
                        ${JFS_SOURCE_DIR}/cuda/lbm_cuda_kernels.cu)



# Add Library
add_definitions(-DJFS_STATIC)
add_library(jfs STATIC ${JFS_SRC_BASE} ${JFS_SRC_DIFF_OPS} ${JFS_SRC_VIS} ${JFS_SRC_TOP} ${JFS_CUDA_SRC})
set_target_properties(
        jfs
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET jfs PROPERTY CUDA_ARCHITECTURES 86)
set_property(TARGET jfs PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(jfs PRIVATE CUDA::cudart -L${CUDAToolkit_LIBRARY_DIR} -lcudadevrt)