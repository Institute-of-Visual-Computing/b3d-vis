add_library(UnityCUDAInteropLogger INTERFACE)

target_include_directories(UnityCUDAInteropLogger
	INTERFACE
		"include"
)	

target_link_libraries(UnityCUDAInteropLogger
	INTERFACE
		B3D::UnityCUDAInterop::PluginAPI
)

add_library(B3D::UnityCUDAInterop::Logger ALIAS UnityCUDAInteropLogger)
