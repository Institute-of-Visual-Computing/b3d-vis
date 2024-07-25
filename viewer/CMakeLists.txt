find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glad CONFIG REQUIRED)
#find_package(imguizmo CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

add_executable(viewer)

IF (WIN32)
	target_sources(viewer PRIVATE App.rc)
ENDIF()

target_sources(viewer PRIVATE
	main.cpp
	NanoViewer.cpp
	App.h
	App.cpp
	Pass.h
	GLUtils.h
	passes/FullscreenTexturePass.h
	passes/FullscreenTexturePass.cpp
	passes/InfinitGridPass.h
	passes/InfinitGridPass.cpp
	passes/DebugDrawPass.h
	passes/DebugDrawPass.cpp
	DebugDrawList.h
	DebugDrawList.cpp
	DebugDrawVertex.h
	GizmoHelper.h
	GizmoHelper.cpp
	Camera.h
	CameraManipulator.h
	InteropUtils.h
	views/ServerConnectSettingsView.h
	views/ServerConnectSettingsView.cpp
	views/VolumeView.h
	views/VolumeView.cpp
	views/ServerConnectView.h
	views/ServerConnectView.cpp
	ServerConnect.h
	ServerConnect.cpp
	IdGenerator.h
	GizmoOperationFlags.h
	FontCollection.h
	FontCollection.cpp
	Flags.h
	framework/Dockspace.h
	framework/Dockspace.cpp
	framework/DockableWindowViewBase.h
	framework/DockableWindowViewBase.cpp
	framework/WindowViewBase.h
	framework/WindowViewBase.cpp
	framework/ModalViewBase.h
	framework/ModalViewBase.cpp
	framework/ApplicationContext.h
	framework/ApplicationContext.cpp
	framework/ExtensionBase.h
	framework/RendererExtensionBase.h
	framework/UpdatableComponentBase.h
	framework/UpdatableComponentBase.cpp
	framework/MenuBar.h
	framework/MenuBar.cpp
	features/transferMapping/TransferMapping.h
	features/transferMapping/TransferMapping.cpp
	features/transferMapping/TransferMappingController.h
	features/transferMapping/TransferMappingController.cpp
	features/transferMapping/TransferMappingView.h
	features/transferMapping/TransferMappingView.cpp
	features/projectExplorer/ProjectExplorer.h
	features/projectExplorer/ProjectExplorer.cpp
	features/projectExplorer/ProjectSelectionView.h
	features/projectExplorer/ProjectSelectionView.cpp
)

target_include_directories(viewer PUBLIC /)

target_compile_features(viewer PUBLIC cxx_std_23)
target_link_libraries(viewer
  PRIVATE
	glfw
	glm::glm
	imgui::imgui
	B3D::Renderer::Library
	B3D::Renderer::Null
	B3D::Renderer::CudaSurfaceObjectWriteTest
	B3D::Renderer::SimpleTriangles
	B3D::Renderer::Nano
	B3D::Renderer::FastVoxelTraversal
	B3D::Renderer::NanoOutOfCore
	OpenGL::GL
	glad::glad
	imguizmo_modefied
	iconFontCppHeaders
	#imguizmo::imguizmo
	B3D::Tools::ColorMap
	${CUDA_nvml_LIBRARY}
	imcurve_modefied
	imspinner
)

option(ENABLE_NANO_PROFILE "Enables Tracy profiling" OFF)
option(ENABLE_NANO_SANITIZERS "Enables address sanitizers" OFF)

if(ENABLE_NANO_PROFILE)
	find_package(Tracy CONFIG REQUIRED)
	target_link_libraries(viewer PRIVATE Tracy::TracyClient)
	target_compile_definitions(viewer PUBLIC -DNANO_PROFILE -DTRACY_ENABLE)
	target_sources(viewer PRIVATE
	main.cpp
	)
endif()

if(ENABLE_NANO_SANITIZERS)
	#target_compile_options(viewer PRIVATE -fsanitize=fuzzer)
	#target_link_options(viewer PRIVATE -fsanitize=fuzzer)
	target_compile_options(viewer PRIVATE -fsanitize=address)
	target_link_options(viewer PRIVATE -fsanitize=address)
	add_definitions(-D_DISABLE_VECTOR_ANNOTATION)
	add_definitions(-D_DISABLE_STRING_ANNOTATION)
	#target_compile_definitions(viewer PRIVATE -D_CRTDBG_MAP_ALLOC)
endif()

target_include_directories(viewer
	PRIVATE
		${Stb_INCLUDE_DIR}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/resources")
	foreach(RESOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")
		list(APPEND commands
			COMMAND ${CMAKE_COMMAND} -E copy_directory 
                ${RESOURCE_DIR}
                "${CMAKE_CURRENT_BINARY_DIR}/resources")
	endforeach()
endif()

add_custom_target(copy_viewer_resources ALL COMMAND ${commands})

add_dependencies(viewer copy_viewer_resources)
