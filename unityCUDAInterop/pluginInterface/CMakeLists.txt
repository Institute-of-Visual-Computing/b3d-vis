set(MODULE_NAME UnityCUDAInteropPluginInterface)

add_library(${MODULE_NAME} INTERFACE)


target_include_directories(${MODULE_NAME} INTERFACE "include")

target_sources(${MODULE_NAME}
	INTERFACE
		src/unity_plugin.cpp
)

set_target_properties(${MODULE_NAME} PROPERTIES CXX_STANDARD 20)


set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "UnityCudaInterop")

add_library(B3D::UnityCUDAInterop::PluginInterface ALIAS ${MODULE_NAME})
