
# opencv
find_package(OpenCV REQUIRED)
list(APPEND icp_inc ${OpenCV_INCLUDE_DIRS})
list(APPEND icp_lib ${OpenCV_LIBS})


if(USE_CUDA)
# cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 --default-stream per-thread -Xcompiler -fopenmp")
list(APPEND icp_inc ${CUDA_INCLUDE_DIRS})
list(APPEND icp_lib ${CUDA_LIBRARIES})
endif()


# eigen
#list(append CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/eigen3/x64/share/eigen3")
set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/eigen3/x64/share/eigen3")
find_package(Eigen3 CONFIG REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


# src
SET(icp_cuda_srcs  icp.cu scene/common.cu scene/edge_scene/edge_scene.cu scene/kdtree_scene/kdtree_scene.cu)
SET(icp_srcs  icp.cpp scene/common.cpp scene/edge_scene/edge_scene.cpp scene/kdtree_scene/kdtree_scene.cpp)


if(USE_CUDA)
CUDA_COMPILE(icp_cuda_objs ${icp_cuda_srcs})
endif()

# lib & test exe
add_library(cuda_icp
               ${icp_srcs}
               ${icp_cuda_srcs}
               ${icp_cuda_objs} 
)
target_include_directories(cuda_icp PUBLIC ${icp_inc})
target_link_libraries(cuda_icp PUBLIC ${icp_lib})
