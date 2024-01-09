set(MODULE_NAME ActionConcrete)
add_library(${MODULE_NAME} SHARED
	ActionConcrete.cpp
)

set_target_properties(${MODULE_NAME} PROPERTIES CXX_STANDARD 20)
target_link_libraries(${MODULE_NAME}
	PRIVATE
		B3D::UnityCUDAInterop::PluginInterface
		B3D::UnityCUDAInterop::ActionInterface
)

set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "UnityCudaInterop/Actions")

install(TARGETS ${MODULE_NAME} DESTINATION ${UNITY_PROJECT_ROOT_DIR}/Assets/${UNITY_ASSET_SUBDIRECTORY_INTEROP_ACTION_SAMPLES}/${MODULE_NAME})
