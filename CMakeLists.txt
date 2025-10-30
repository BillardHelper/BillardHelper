cmake_minimum_required(VERSION 3.10.0)
project(BilliardHelper VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(BilliardHelper main.cpp)
target_link_libraries(BilliardHelper ${OpenCV_LIBS})