cmake_minimum_required(VERSION 3.0)
project(example LANGUAGES CXX)

add_executable(reading_keras_model reading_keras_model.cpp)
target_link_libraries(reading_keras_model PUBLIC tensor_graph)