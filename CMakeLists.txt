cmake_minimum_required(VERSION 3.13.0)
project(jfs VERSION 0.1.0 LANGUAGES CUDA CXX)

find_package(CUDA)

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

# INSTALL PATH:
#   E.g. if /usr/local will install headers in /usr/local/include/jfs
#   and library in /usr/local/lib/
#
#   Note that CMake doesn't have an uninstall feature
#   To uninstall, manual removal of the files is necessary
    
    set(INSTALL_PATH ${PROJECT_SOURCE_DIR}/install)

# END CUSTOM OPTIONS
#####################################################################

include_directories(${PROJECT_SOURCE_DIR}
                    ${EIGEN_PATH})

set(JFS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/jfs)

# Header Files
set(JFS_HEADERS_BASE    ${JFS_SOURCE_DIR}/base/grid2D.h
                        ${JFS_SOURCE_DIR}/base/grid3D.h)

set(JFS_HEADERS_DIFF_OPS    ${JFS_SOURCE_DIR}/differential_ops/grid_diff2d.h
                            ${JFS_SOURCE_DIR}/differential_ops/grid_diff3d.h)

set(JFS_HEADERS_VIS     ${JFS_SOURCE_DIR}/visualization/grid_smoke2d.h
                        ${JFS_SOURCE_DIR}/visualization/grid_smoke3d.h)

set(JFS_HEADERS_TOP     ${JFS_SOURCE_DIR}/static_build/jfs_inline.h
                        ${JFS_SOURCE_DIR}/JSSFSolver.h
                        ${JFS_SOURCE_DIR}/JSSFSolver3D.h
                        ${JFS_SOURCE_DIR}/LBMSolver.h)


# Source Files
set(JFS_SRC_BASE    ${JFS_SOURCE_DIR}/base/grid2D.cpp
                    ${JFS_SOURCE_DIR}/base/grid3D.cpp)

set(JFS_SRC_DIFF_OPS    ${JFS_SOURCE_DIR}/differential_ops/grid_diff2d.cpp
                        ${JFS_SOURCE_DIR}/differential_ops/grid_diff3d.cpp)

set(JFS_HEADERS_VIS     ${JFS_SOURCE_DIR}/visualization/grid_smoke2d.cpp
                        ${JFS_SOURCE_DIR}/visualization/grid_smoke3d.cpp)

set(JFS_SRC_TOP ${JFS_SOURCE_DIR}/JSSFSolver.cpp
                ${JFS_SOURCE_DIR}/JSSFSolver3D.cpp
                ${JFS_SOURCE_DIR}/LBMSolver.cpp)

# CUDA Header Files
set(JFS_CUDA_HEADERS    ${JFS_SOURCE_DIR}/cuda/grid/cuda_grid2d.h
                        ${JFS_SOURCE_DIR}/cuda/lbm_solver_cuda.h
                        ${JFS_SOURCE_DIR}/cuda/lbm_cuda_kernels.h)

# CUDA Source Files
set(JFS_CUDA_SRC        ${JFS_SOURCE_DIR}/cuda/grid/cuda_grid2d.cu
                        ${JFS_SOURCE_DIR}/cuda/lbm_solver_cuda.cu
                        ${JFS_SOURCE_DIR}/cuda/lbm_cuda_kernels.cu)



# Add Library
add_definitions(-DJFS_STATIC)
add_library(jfs STATIC ${JFS_SRC_BASE} ${JFS_SRC_DIFF_OPS} ${JFS_HEADERS_VIS} ${JFS_SRC_TOP} ${JFS_CUDA_SRC})
set_property(TARGET jfs PROPERTY CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -gencode=arch=compute_86,code=sm_86")
set_target_properties(jfs PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(jfs PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY})

# Installation of jfs files
install(TARGETS jfs DESTINATION ${INSTALL_PATH}/lib)
install(FILES ${JFS_HEADERS_TOP} DESTINATION ${INSTALL_PATH}/include/jfs)
install(FILES ${JFS_HEADERS_BASE} DESTINATION ${INSTALL_PATH}/include/jfs/base)
install(FILES ${JFS_HEADERS_DIFF_OPS} DESTINATION ${INSTALL_PATH}/include/jfs/differential_ops)
