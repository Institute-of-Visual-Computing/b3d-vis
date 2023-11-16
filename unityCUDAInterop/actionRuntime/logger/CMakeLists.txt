add_library(UnityCUDAInteropLogger INTERFACE)

target_include_directories(UnityCUDAInteropLogger
	INTERFACE
		"include"
)	

target_link_libraries(UnityCUDAInteropLogger
	INTERFACE
		UnityCUDAInterop::UnityPluginAPI
)

add_library(UnityCUDAInterop::Logger ALIAS UnityCUDAInteropLogger)
