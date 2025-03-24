
set(MODULE_NAME "B3DRendererNull")
set(SRC NullRenderer.cpp)

add_library(${MODULE_NAME} ${SRC})
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_20)
target_include_directories(${MODULE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${MODULE_NAME} PRIVATE B3D::Renderer::Framework)
add_library(B3D::Renderer::Null ALIAS ${MODULE_NAME})

set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "Renderer")
