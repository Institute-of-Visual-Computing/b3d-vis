embed_ptx(
  OUTPUT_TARGET
    "SimpleTriangles-ptx"
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    SimpleTrianglesDeviceCode.cu
)

find_package(imgui CONFIG REQUIRED)

add_b3d_renderer(
	NAME
		SimpleTriangles
	SOURCES
		simpleTrianglesHostCode.cpp
	HEADERS
		deviceCode.h
	LIBRARIES
		"SimpleTriangles-ptx"
		imgui::imgui
)

