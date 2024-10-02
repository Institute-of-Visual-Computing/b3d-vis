set(MODULE_NAME "ActionSimpleTriangles")
add_library(${MODULE_NAME} SHARED
	"src/ActionSimpleTriangles.cpp"
)

target_link_libraries(${MODULE_NAME}
	PRIVATE
		B3D::UnityCUDAInterop::PluginInterface
		B3D::UnityCUDAInterop::ActionInterface
		B3D::Renderer::Library
		B3D::Renderer::SimpleTriangles
)


set_target_properties(${MODULE_NAME} PROPERTIES CXX_STANDARD 20)

set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "UnityCudaInterop/Actions")

target_compile_definitions(${MODULE_NAME} PUBLIC STB_IMAGE_IMPLEMENTATION)

install(TARGETS ${MODULE_NAME} DESTINATION ${UNITY_PROJECT_ROOT_DIR}/Assets/${UNITY_ASSET_SUBDIRECTORY_INTEROP_ACTION_SAMPLES}/${MODULE_NAME})
