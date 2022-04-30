cmake_minimum_required(VERSION 3.22)

project(minnet)

set(CXX_STANDARD 2A)
set(CXX_STANDARD_REQUIRE TRUE)

# find openmp
find_package(OpenMP)

# set opencv
set (OpenCV_DIR D:\\opencv\\build)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "	config: ${OpenCV_DIR}")
message(STATUS "	version: ${OpenCV_VERSION}")
message(STATUS "	librarier: ${OpenCV_LIBS}")
message(STATUS "	include path: ${OpenCV_INCLUDE_DIRS}")

# add src and opencv include path
include_directories(${OpenCV_INCLUDE_DIRS} include)

add_subdirectory(src)
add_subdirectory(example)