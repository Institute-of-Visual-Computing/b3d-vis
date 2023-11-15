add_library(UnityCUDAInteropRuntime SHARED
	"src/PluginHandler.cpp"
	"src/unity_plugin.cpp"
	"src/RenderAPI/RenderAPI.cpp"
	"src/RenderAPI/RenderAPI_D3D11.cpp"
	"src/PluginLogger.cpp"
	"src/Texture/Texture_D3D11.cpp" "src/Texture/Texture.cpp")

target_include_directories(UnityCUDAInteropRuntime 
	PUBLIC
		"include"
	PRIVATE
		"src/include"
		"interface"
	INTERFACE
		"interface"
)

target_link_libraries(UnityCUDAInteropRuntime
	PRIVATE
		UnityCUDAInterop::Action
		UnityCUDAInterop::UnityPluginAPI
		UnityCUDAInterop::Logger
		d3d11.lib
	PUBLIC
        CUDA::cudart
)

set_target_properties(UnityCUDAInteropRuntime PROPERTIES CXX_STANDARD 20)

add_library(UnityCUDAInterop::Runtime ALIAS UnityCUDAInteropRuntime)

# install(TARGETS UnityCUDAInteropRuntime DESTINATION ${UNITY_PROJECT_ROOT_DIR}/Assets/${UNITY_ASSETS_SUBDIRECTORY_RUNTIME})
