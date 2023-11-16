add_library(UnityCUDAInteropAction INTERFACE)

target_link_libraries(UnityCUDAInteropAction
	INTERFACE
		UnityCUDAInterop::UnityPluginAPI
		UnityCUDAInterop::Runtime
		UnityCUDAInterop::Logger
)

target_include_directories(UnityCUDAInteropAction
	INTERFACE
		"include"
)	

add_library(UnityCUDAInterop::Action ALIAS UnityCUDAInteropAction)
