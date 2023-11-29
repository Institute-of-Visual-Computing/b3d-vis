set(MODULE_NAME "B3DRendererCudaSurfaceObjectWriteTest")
set(SRC cudaSurfaceObjectWriteTestRenderer.cu)

add_library(${MODULE_NAME} ${SRC})
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)
target_include_directories(${MODULE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${MODULE_NAME}
	PRIVATE
		B3D::Renderer::Library
)
set_property(TARGET ${MODULE_NAME} PROPERTY CUDA_STANDARD 17)



add_library(B3D::Renderer::CudaSurfaceObjectWriteTest ALIAS ${MODULE_NAME})


set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "Renderer")
