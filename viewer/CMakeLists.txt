find_package(imgui CONFIG REQUIRED)

add_executable(viewer main.cpp NanoViewer.cpp)
target_compile_features(viewer PUBLIC cxx_std_20)
target_link_libraries(viewer
  PRIVATE
	owl_viewer
	owl::owl
	glfw
	imgui::imgui
	b3d::null
	b3d::lib
)
