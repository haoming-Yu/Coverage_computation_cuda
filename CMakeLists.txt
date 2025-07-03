cmake_minimum_required(VERSION 3.18)
project(COVERAGE CUDA CXX C)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA setting
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# set opencv
find_package(OpenCV REQUIRED)

# set include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# sources
set(SOURCES
    src/depth.cpp
    src/mesh.cpp
    src/rply.c
    src/main.cpp
    src/camera.cpp
    src/cudaCoverage.cu
)

# headers
set(HEADERS
    include/depth.h
    include/rply.h
    include/rplyfile.h
    include/mesh.h
    include/camera.h
    include/cudaUtils.cuh
    include/cudaCoverage.cuh
)

# create executable file
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# set CUDA architectures
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "72"  # adjust according to the architecture of GPUs
)

# link cuda lib
target_link_libraries(${PROJECT_NAME}
    stdc++fs
    cuda
    cudart
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
)

# compile options
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
