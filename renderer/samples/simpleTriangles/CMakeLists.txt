
find_package(Stb REQUIRED)

embed_ptx(
  OUTPUT_TARGET
    SimpleTriangles-ptx
  PTX_LINK_LIBRARIES
    owl::owl
	B3D::Renderer::Framework
  SOURCES
    SimpleTrianglesDeviceCode.cu
)

set_property(TARGET SimpleTriangles-ptx PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET SimpleTriangles-ptx_ptx PROPERTY FOLDER "Renderer/PTX Targets")

find_package(imgui CONFIG REQUIRED)

add_b3d_renderer(
	NAME
		SimpleTriangles
	SOURCES
		simpleTrianglesHostCode.cpp
	HEADERS
		deviceCode.h
	LIBRARIES
		B3D::Tools::ColorMap
		SimpleTriangles-ptx
		imgui::imgui
)

target_include_directories(SimpleTriangles
	PRIVATE
		${Stb_INCLUDE_DIR}
)

