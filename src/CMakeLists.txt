# find openmp
# add include path
include_directories(${ROOT_DIR}/include)

aux_source_directory(. DIR_LIB_SRCS)
add_library(MINNET STATIC ${DIR_LIB_SRCS})

if (DEFINED linux)
    set(SUFFIX .a)
endif()
if (DEFINED windows)
    set(SUFFIX .lib)
endif()

set_target_properties(MINNET PROPERTIES
        OUTPUT_NAME minnet
        PREFIX lib_
        SUFFIX ${SUFFIX}
        VERSION ${CMAKE_PROJECT_VERSION}
        SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(MINNET PUBLIC OpenMP::OpenMP_CXX)
endif()