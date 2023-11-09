
set(module_name "nullRenderer")
set(src NullRenderer.cpp)

add_library(${module_name} ${src})
target_compile_features(${module_name} PUBLIC cxx_std_20)
target_include_directories(${module_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${module_name} PRIVATE b3d::lib)
add_library(b3d::null ALIAS ${module_name})
