set(MODULE_NAME "ActionFitsNvdbRenderer")
add_library(${MODULE_NAME} SHARED)

target_sources(
	${MODULE_NAME}
PRIVATE
	src/ActionFitsNvdbRenderer.cpp
)

target_link_libraries(${MODULE_NAME}
	PRIVATE
		B3D::UnityCUDAInterop::PluginInterface
		B3D::UnityCUDAInterop::ActionInterface
		B3D::Renderer::Framework
		# Replace the next line with the renderer you would like to bring to Unity
		B3D::Renderer::FitsNvdb
		B3D::Header::Shared::NvdbRendering
		B3D::Tools::RuntimeDataset
)

set_target_properties(${MODULE_NAME} PROPERTIES CXX_STANDARD 20)

set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "UnityCudaInterop/Actions")
target_compile_options(${MODULE_NAME} PRIVATE /W4 /WX /external:W0 /external:anglebrackets)

# Unity Actions directory
if(UNITY_PROJECT_ROOT_DIR)
	install(TARGETS ${MODULE_NAME} DESTINATION ${UNITY_PROJECT_ROOT_DIR}/Assets/${UNITY_ASSET_SUBDIRECTORY_INTEROP_ACTIONS}/${MODULE_NAME})
endif()
