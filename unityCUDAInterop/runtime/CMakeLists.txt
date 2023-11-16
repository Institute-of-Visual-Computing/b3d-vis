set(MODULE_NAME UnityCUDAInteropRuntime)

add_library(${MODULE_NAME} SHARED
	"src/PluginHandler.cpp"
	"src/unity_plugin.cpp"
	"src/RenderAPI/RenderAPI.cpp"
	"src/RenderAPI/RenderAPI_D3D11.cpp"
	"src/PluginLogger.cpp"
	"src/Texture/Texture_D3D11.cpp" "src/Texture/Texture.cpp")

target_include_directories(${MODULE_NAME} 
	PUBLIC
		"include"
	PRIVATE
		"src/include"
		"interface"
	INTERFACE
		"interface"
)

target_link_libraries(${MODULE_NAME}
	PRIVATE
		B3D::UnityCUDAInterop::ActionInterface
		B3D::UnityCUDAInterop::PluginAPI
		B3D::UnityCUDAInterop::Logger
		d3d11.lib
	PUBLIC
        CUDA::cudart
)

set_target_properties(${MODULE_NAME} PROPERTIES CXX_STANDARD 20)

add_library(B3D::UnityCUDAInterop::Runtime ALIAS ${MODULE_NAME})

set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "UnityCudaInterop")
# install(TARGETS ${MODULE_NAME} DESTINATION ${UNITY_PROJECT_ROOT_DIR}/Assets/${UNITY_ASSETS_SUBDIRECTORY_RUNTIME})
