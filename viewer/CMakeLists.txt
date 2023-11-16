find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(VulkanHeaders CONFIG)

add_executable(viewer main.cpp NanoViewer.cpp)
target_compile_features(viewer PUBLIC cxx_std_20)
target_link_libraries(viewer
  PRIVATE
	owl_viewer
	owl::owl
	glfw
	imgui::imgui
	B3D::Renderer::Library
	B3D::Renderer::Null
	B3D::Renderer::FastVoxelTraversal
	OpenGL::GL
	glad::glad
	Vulkan::Vulkan
	Vulkan::Headers
)
