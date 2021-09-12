# JFS
JFluidSolvers: c++ library of various fluid simulation methods.

It used to have header-only functionality but it's too difficult to maintain. There are two main ways of using the library. Building it and linking it manually or including at as a subdirectory in a cmake file.

## Dependencies
All the dependencies are included as git submodules in the `extern` folder of this repository.

- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page):  If you're project already includes Eigen, you will 
  need to change their locations in the [CMakeLists.txt](./CMakeLists.txt).
- [CUDA 11+](https://developer.nvidia.com/cuda-toolkit)
- [CMake](https://cmake.org/) - not a submodule, just needed to build the library


## Building the Library

### Windows
```
git clone --recursive https://github.com/jackm97/JFS
cd JFS
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Unix
```
git clone --recursive https://github.com/jackm97/JFS
cd JFS
mkdir build
cd build
cmake ..
make
```

Once built the `jfs` library can be linked with any project. Make sure the repository is in the include path along 
with all dependencies. Note that the CUDA runtime libraries also need to be linked with the final application.

## CMake Subdirectory
To include the jfs library in a CMake project just include the following in the top-level `CMakeLists.txt`:

```
#########################LIBJFS########################
# SET SOME OPTIONS FOR LIBJFS
#   JFS PATH
set(PATH_TO_JFS ${PROJECT_SOURCE_DIR}/extern/JFS/)
#   EXTERNAL LIB PATHS
set(EIGEN_PATH ${PROJECT_SOURCE_DIR}/extern/JFS/extern/eigen)

add_definitions(-DJFS_STATIC)
add_subdirectory(${PATH_TO_JFS})

include_directories(${PATH_TO_JFS}
                    ${EIGEN_PATH})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_definitions(-DUSE_OPENMP)
    target_link_libraries(jfs PUBLIC OpenMP::OpenMP_CXX)
endif()
#########################LIBJFS########################
```

