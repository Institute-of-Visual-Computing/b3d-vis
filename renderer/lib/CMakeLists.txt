set(module_name "lib")
set(src Logging.cpp RendererBase.cpp)
set(shared_headers Logging.h RendererBase.h)

add_library(${module_name} ${src} ${shared_headers})
target_include_directories(${module_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${module_name} PUBLIC cxx_std_20)
target_link_libraries(${module_name}
  PRIVATE
    owl::owl
)
add_library(b3d::lib ALIAS ${module_name})
