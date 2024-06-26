set(MODULE_NAME UnityCUDAInteropLoggerInterface)
add_library(UnityCUDAInteropLoggerInterface INTERFACE)

target_include_directories(${MODULE_NAME}
	INTERFACE
		"include"
)	

target_link_libraries(${MODULE_NAME}
	INTERFACE
		B3D::UnityCUDAInterop::PluginAPI
)

add_library(B3D::UnityCUDAInterop::LoggerInterface ALIAS ${MODULE_NAME})
