find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(VulkanHeaders CONFIG)

#include(embed_ptx)
embed_ptx(
  OUTPUT_TARGET
    lib-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    test.cu
)

add_executable(viewer main.cpp NanoViewer.cpp Vulkan.h Vulkan.cpp Viewer.h Viewer.cpp App.h App.cpp test.h)
target_compile_features(viewer PUBLIC cxx_std_20)
target_link_libraries(viewer
  PRIVATE
	lib-ptx
	owl_viewer
	owl::owl
	glfw
	imgui::imgui
	B3D::Renderer::Library
	B3D::Renderer::Null
	B3D::Renderer::CudaSurfaceObjectWriteTest
	B3D::Renderer::Nano
	OpenGL::GL
	glad::glad
	Vulkan::Vulkan
	Vulkan::Headers
)
