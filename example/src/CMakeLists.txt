include_directories(${OpenCV_INCLUDE_DIRS} ${EXAMPLE_DIR}/include)

aux_source_directory(. DIR_LIB_SRCS)
add_library(DATASET ${DIR_LIB_SRCS})