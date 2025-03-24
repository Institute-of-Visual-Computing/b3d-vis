set(MODULE_NAME UnityCUDAInteropHelper)

add_library(${MODULE_NAME} STATIC
	src/Texture/Texture.cpp
	src/Texture/Texture_D3D11.cpp
    src/PluginHandler.cpp
	src/PluginLogger.cpp
	src/RenderAPI/RenderAPI.cpp
	src/RenderAPI/RenderAPI_D3D11.cpp
	src/RenderingContext/RenderingContext_D3D11.cpp
	src/SyncPrimitive/SyncPrimitive_D3D11.cpp)

# If Shared
# set_property(TARGET ${MODULE_NAME} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS true)


target_include_directories(${MODULE_NAME} 
	PUBLIC
		include
	PRIVATE
		src/include
)

set_target_properties(${MODULE_NAME} PROPERTIES CXX_STANDARD 20)

find_package(CUDAToolkit)

target_link_libraries(${MODULE_NAME}
	PRIVATE
		d3d11.lib
		B3D::UnityCUDAInterop::ActionInterface
	PUBLIC
		B3D::UnityCUDAInterop::PluginAPI
		B3D::UnityCUDAInterop::LoggerInterface
		CUDA::cudart_static
)

set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "UnityCudaInterop")

add_library(B3D::UnityCUDAInterop::Helper ALIAS ${MODULE_NAME})
