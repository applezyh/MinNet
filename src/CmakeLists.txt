aux_source_directory(. DIR_LIB_SRCS)
include_directories(${DIR_LIB_SRCS}/../include)
add_library(MINNET ${DIR_LIB_SRCS})