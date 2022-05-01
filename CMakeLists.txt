cmake_minimum_required(VERSION 3.0)

project(minnet)

# adapt operation system
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
	set(linux ON)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(windows ON)
endif()

# set OpenMP
find_package(OpenMP)

# set opencv
set(OpenCV_DIR D:\\opencv\\build)
find_package(OpenCV)
if(OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "	config: ${OpenCV_DIR}")
    message(STATUS "	version: ${OpenCV_VERSION}")
    message(STATUS "	librarier: ${OpenCV_LIBS}")
    message(STATUS "	include path: ${OpenCV_INCLUDE_DIRS}")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-w -O2 -fconcepts")

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(src)
if(OpenCV_FOUND)
    add_subdirectory(example)
endif()
