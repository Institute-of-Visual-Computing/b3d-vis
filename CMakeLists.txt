cmake_minimum_required(VERSION 3.26)

project(b3d_visualisation
    VERSION 0.1
    LANGUAGES CXX CUDA
)

set(owl_dir ${PROJECT_SOURCE_DIR}/thirdParty/owl)
OPTION(OWL_BUILD_SAMPLES OFF)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)
add_subdirectory(${owl_dir}/samples/common)
add_subdirectory(viewer)
