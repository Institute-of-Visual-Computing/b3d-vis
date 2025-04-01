find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenVDB CONFIG REQUIRED COMPONENTS nanovdb)
find_package(Stb REQUIRED)

set(viewerTargetName NanoViewer)
add_executable(${viewerTargetName})

IF (WIN32)
	target_sources(${viewerTargetName} PRIVATE App.rc)
ENDIF()

target_sources(${viewerTargetName} PRIVATE
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
	CameraControllers.h
	CameraControllers.cpp
	InteropUtils.h
	views/VolumeView.h
	views/VolumeView.cpp
	views/ServerConnectView.h
	views/ServerConnectView.cpp
	IdGenerator.h
	GizmoOperationFlags.h
	FontCollection.h
	FontCollection.cpp
	Flags.h
	ApplicationSettings.h
	ApplicationSettings.cpp
	GuiDefaultLayoutSettings.h
	GuiDefaultLayoutSettings.cpp
	Animation.h
	GLGpuTimers.h
	ProfilersWindow.h
	ImGuiExtension.h
	ImGuiExtension.cpp
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
	features/serverConnect/ServerConnectView.h
	features/serverConnect/ServerConnectView.cpp
	features/serverConnect/ServerConnectSettingsView.h
	features/serverConnect/ServerConnectSettingsView.cpp
	features/serverConnect/ServerAddEditView.h
	features/serverConnect/ServerAddEditView.cpp
	features/projectExplorer/ProjectExplorerView.h
	features/projectExplorer/ProjectExplorerView.cpp
	features/projectExplorer/ProjectExplorerController.cpp
	features/projectExplorer/ProjectExplorerController.h
	features/projectExplorer/SofiaParameterSummaryView.h
	features/projectExplorer/SofiaParameterSummaryView.cpp
	features/projectExplorer/AddNewProjectView.h
	features/projectExplorer/AddNewProjectView.cpp
	features/projectExplorer/DeleteProjectView.h
	features/projectExplorer/DeleteProjectView.cpp
	features/projectExplorer/EditProjectView.h
	features/projectExplorer/EditProjectView.cpp
	features/sofiaSearch/SoFiaSearch.h
	features/sofiaSearch/SoFiaSearch.cpp
	features/sofiaSearch/SoFiaSearchView.h
	features/sofiaSearch/SoFiaSearchView.cpp
	features/sofiaSearch/ParamsSerializer.h
)

target_include_directories(${viewerTargetName} PUBLIC /)
target_compile_features(${viewerTargetName} PUBLIC cxx_std_23)
target_link_libraries(${viewerTargetName}
  PRIVATE
	glfw
	glm::glm
	imgui::imgui
	B3D::Renderer::Framework
	B3D::Renderer::Null
	B3D::Renderer::FitsNvdb
	B3D::Tools::ServerClient
	B3D::Tools::ServerFileProvider
	B3D::Tools::Project
	B3D::Tools::RuntimeDataset
	B3D::Tools::Fits
	OpenGL::GL
	glad::glad
	imguizmo_modefied
	iconFontCppHeaders
	B3D::Tools::ColorMap
	${CUDA_nvml_LIBRARY}
	imcurve_modefied
	imspinner
	OpenVDB::nanovdb
	nlohmann_json::nlohmann_json
	legit
	B3D::Header::Sofia
	imGuiFileDialog
)

target_include_directories(${viewerTargetName} PRIVATE ${Stb_INCLUDE_DIR})

target_compile_definitions(${viewerTargetName} PUBLIC B3D_USE_NLOHMANN_JSON)

if(ENABLE_NANO_VIEWER_PROFILE)
	find_package(Tracy CONFIG REQUIRED)
	target_link_libraries(${viewerTargetName} PRIVATE Tracy::TracyClient)
	target_compile_definitions(${viewerTargetName} PUBLIC -DNANO_PROFILE -DTRACY_ENABLE)
	target_sources(${viewerTargetName} PRIVATE
	main.cpp
	)
endif()

if(ENABLE_NANO_VIEWER_SANITIZERS)
	target_compile_options(${viewerTargetName}  PRIVATE -fsanitize=address)
	target_link_options(${viewerTargetName}  PRIVATE -fsanitize=address)
	add_definitions(-D_DISABLE_VECTOR_ANNOTATION)
	add_definitions(-D_DISABLE_STRING_ANNOTATION)
endif()

target_include_directories(${viewerTargetName}
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
set_property(TARGET copy_viewer_resources PROPERTY FOLDER "Utility")

add_dependencies(${viewerTargetName}  copy_viewer_resources)

if(MSVC)
    target_compile_options(${viewerTargetName} PRIVATE /W4 /WX /external:W0 /external:anglebrackets)
endif()
