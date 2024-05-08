cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 17)
project(find_shape_model)

# arm or x86
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    SET(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
ELSE()
    #SET(PLATFORM_COMPILE_FLAGS "-march=native")
    #please check you cpu 
    SET(PLATFORM_COMPILE_FLAGS "-arch:AVX2")

    # some places of the algorithm are designed for 128 SIMD
    # so 128 SSE may slightly faster than 256 AVX, you may want this
    #SET(PLATFORM_COMPILE_FLAGS "-msse ")  # SSE only -msse2 -msse3 -msse4 -mssse3
ENDIF()

#SET(PLATFORM_COMPILE_FLAGS "-DMIPP_NO_INTRINSICS")  # close SIMD
SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
#compiler option
IF(MSVC)
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
ELSE()
    SET(COMMON_COMPILE_FLAGS "-fopenmp") # -fno-strict-aliasing
    SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
ENDIF()

# opencv
list(APPEND CMAKE_PREFIX_PATH "$ENV{OpenCV_Path}/x64/vc15/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# include MIPP headers
include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")

# icp for refine
option(USE_CUDA "use cuda or not" OFF)

if(USE_CUDA)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0)
add_definitions(-DCUDA_ON)
endif()

add_subdirectory(cuda_icp)

# compile to static lib
set(LIB_SRC line2Dup.cpp find_shape_model.cpp)
add_library(findShapeModel STATIC ${LIB_SRC} )
target_link_libraries(findShapeModel ${OpenCV_LIBS})
target_link_libraries(findShapeModel cuda_icp)

# test use lib exe
add_executable(${PROJECT_NAME}_test "test.cpp")
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME}_test findShapeModel)


