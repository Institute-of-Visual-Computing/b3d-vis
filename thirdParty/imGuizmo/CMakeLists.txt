find_package(imgui CONFIG REQUIRED)

add_library(imguizmo_modefied STATIC)
target_sources(imguizmo_modefied PUBLIC
	ImGuizmo.h
	ImGuizmo.cpp
)

target_include_directories(imguizmo_modefied PUBLIC /)

target_compile_features(imguizmo_modefied PUBLIC cxx_std_23)
target_link_libraries(imguizmo_modefied 
  PRIVATE
	imgui::imgui
)
