set(MODULE_NAME "B3DRendererLibrary")

set(SRC
	Logging.cpp
	RendererBase.cpp
	RenderData.cpp
)

set(SHARED_HEADERS
	Logging.h
	RendererBase.h
	DebugDrawListBase.h
	GizmoHelperBase.h
	NullDebugDrawList.h
	NullGizmoHelper.h
	OwlExtensions.h
	Common.h
	RenderFeature.h
	ProfilerResult.h
	CudaGpuTimers.h
	Profiler.h
	Profiler.cpp
	features/ColorMapFeature.h
	features/ColorMapFeature.cpp
	features/TransferFunctionFeature.h
	features/TransferFunctionFeature.cpp
	features/RenderSyncFeature.cpp
	features/RenderSyncFeature.h
	features/BackgroundColorFeature.cpp
	features/BackgroundColorFeature.h
	features/RenderTargetFeature.cpp
	features/RenderTargetFeature.h
	features/SoFiASubregionFeature.cpp
	features/SoFiASubregionFeature.h
)
find_package(httplib CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
add_library(${MODULE_NAME} ${SRC} ${SHARED_HEADERS})
target_include_directories(${MODULE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_20)
target_link_libraries(${MODULE_NAME}
  PUBLIC
	owl::owl
	imgui::imgui
	legit_task
  PRIVATE
	httplib::httplib
)
add_library(B3D::Renderer::Library ALIAS ${MODULE_NAME})

set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "Renderer")
