add_library(UnityCUDAInteropActionInterface INTERFACE)

target_link_libraries(UnityCUDAInteropActionInterface
	INTERFACE
		B3D::UnityCUDAInterop::PluginAPI
		B3D::UnityCUDAInterop::Runtime
		B3D::UnityCUDAInterop::Logger
)

target_include_directories(UnityCUDAInteropActionInterface
	INTERFACE
		"include"
)	

add_library(B3D::UnityCUDAInterop::ActionInterface ALIAS UnityCUDAInteropActionInterface)
