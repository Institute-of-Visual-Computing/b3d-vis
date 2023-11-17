set(MODULE_NAME "ActionTest")
add_library(${MODULE_NAME} SHARED
	"src/ActionTest.cpp"
)

target_link_libraries(${MODULE_NAME}
	PRIVATE
		B3D::UnityCUDAInterop::ActionInterface
)

set_target_properties(${MODULE_NAME} PROPERTIES CXX_STANDARD 20)

set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "UnityCudaInterop/Actions")

# install(TARGETS ${MODULE_NAME} DESTINATION ${UNITY_PROJECT_ROOT_DIR}/Assets/${UNITY_ASSETS_SUBDIRECTORY_ACTIONS}/${MODULE_NAME})
