add_executable( minnet_demo MinNet.cpp )
target_link_libraries( minnet_demo MINNET )
target_link_libraries( minnet_demo ${OpenCV_LIBS} )
if(OpenMP_CXX_FOUND)
    message(STATUS "find OpenMP. use OpenMP Parallelization program")
    target_link_libraries( minnet_demo OpenMP::OpenMP_CXX )
endif()

