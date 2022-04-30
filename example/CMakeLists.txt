set(EXAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(src)
include_directories(${OpenCV_INCLUDE_DIRS} include ${ROOT_DIR}/include)

add_executable(demo MinNet.cpp)
target_link_libraries(demo ${OpenCV_LIBRARIES} DATASET MINNET)