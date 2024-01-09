set(MODULE_NAME UnityCUDAInteropActionInterface)

add_library(${MODULE_NAME} INTERFACE)

target_include_directories(${MODULE_NAME}
	INTERFACE
		"include"
)

target_link_libraries(${MODULE_NAME}
	INTERFACE
		B3D::UnityCUDAInterop::Helper
)

set_target_properties(${MODULE_NAME} PROPERTIES CXX_STANDARD 20)

add_library(B3D::UnityCUDAInterop::ActionInterface ALIAS ${MODULE_NAME})
