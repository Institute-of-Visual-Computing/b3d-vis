find_package(imgui CONFIG REQUIRED)

add_library(imguizmo_modefied STATIC)
target_sources(
	imguizmo_modefied
PRIVATE
	ImGuizmo.h
	ImGuizmo.cpp
)

target_include_directories(imguizmo_modefied PUBLIC /)

target_compile_features(imguizmo_modefied PUBLIC cxx_std_23)
target_link_libraries(imguizmo_modefied 
  PRIVATE
	imgui::imgui
)

set_property(TARGET imguizmo_modefied PROPERTY FOLDER "ThirdParty/imguizmo_modefied")
if(MSVC)
    target_compile_options(imguizmo_modefied PRIVATE /W0)
endif()
