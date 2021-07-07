cmake_minimum_required(VERSION 3.0)
project(tensor_graph VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse2 -mavx")

set(CMAKE_MODULE_PATH 
${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(TensorFlow REQUIRED)

add_library(${PROJECT_NAME} 
SHARED 
  src/tensor_graph.cpp
)

target_include_directories(${PROJECT_NAME}
PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${TF_INCLUDE_DIRS}
)

target_compile_features(${PROJECT_NAME}
PUBLIC
  cxx_std_17
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
  ${TF_LIBRARIES}
)

add_subdirectory(example)
