cmake_minimum_required(VERSION 3.0.0)
project(jfs VERSION 0.1.0)

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

add_definitions(-DJFS_STATIC)

add_library(jfs STATIC ${JFS_SOURCE_DIR}/fluid2D.cpp ${JFS_SOURCE_DIR}/JSSFSolver.cpp)

set(JFS_HEADERS ${JFS_SOURCE_DIR}/static_build/jfs_inline.h
                ${JFS_SOURCE_DIR}/JSSFSolver.h
                ${JFS_SOURCE_DIR}/fluid2D.h)

install(TARGETS jfs DESTINATION ${INSTALL_PATH}/lib)
install(FILES ${JFS_HEADERS} DESTINATION ${INSTALL_PATH}/include/jfs)
install(DIRECTORY ${EIGEN_PATH}/Eigen DESTINATION ${INSTALL_PATH}/include/jfs)
