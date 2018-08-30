# Project's name
PROJECT ( LifeCuda C CXX )

# Specify the minimum version for CMake

cmake_minimum_required(VERSION 3.1)

add_definitions(-std=c++11)

find_package(GTest REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDA REQUIRED)

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}/CudaLife/include" "${PROJECT_SOURCE_DIR}/CudaLifeTests/include" ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})

cuda_add_executable(tests 
    "${PROJECT_SOURCE_DIR}/CudaLife/src/CudaLife.cu"
    "${PROJECT_SOURCE_DIR}/CudaLifeTests/src/BenchmarkTests.cpp"
    #"${PROJECT_SOURCE_DIR}/CudaLifeTests/src/CpuLifeTests.cpp"
    "${PROJECT_SOURCE_DIR}/CudaLifeTests/src/GpuLifeTests.cpp"
    "${PROJECT_SOURCE_DIR}/CudaLifeTests/src/main.cpp"
)

cuda_add_executable(gui 
    "${PROJECT_SOURCE_DIR}/CudaLife/src/CudaLife.cu"
    #"${PROJECT_SOURCE_DIR}/CudaLifeTests/src/BenchmarkTests.cpp"
    #"${PROJECT_SOURCE_DIR}/CudaLifeTests/src/CpuLifeTests.cpp"
    #"${PROJECT_SOURCE_DIR}/CudaLifeTests/src/GpuLifeTests.cpp"
    "${PROJECT_SOURCE_DIR}/CudaLifeGui/src/main.cpp"
)

target_link_libraries(tests ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GTEST_LIBRARIES} pthread) 
target_link_libraries(gui ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARY}) 
