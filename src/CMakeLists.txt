# find openmp
# add include path
include_directories(${ROOT_DIR}/include)

aux_source_directory(. DIR_LIB_SRCS)
add_library(MINNET ${DIR_LIB_SRCS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(MINNET PUBLIC OpenMP::OpenMP_CXX)
endif()