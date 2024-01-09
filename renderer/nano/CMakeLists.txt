embed_ptx(
	OUTPUT_TARGET
		nano-ptx
	PTX_LINK_LIBRARIES
		owl::owl
	SOURCES
		NanoRenderer.cu
	#DEVICE_DEBUG
)

find_package(OpenVDB CONFIG REQUIRED COMPONENTS nanovdb)
find_package(imgui CONFIG REQUIRED)

add_b3d_renderer(
	NAME
		Nano
	SOURCES
		NanoRenderer.cpp
	HEADERS
		NanoRenderer.h
	LIBRARIES
		nano-ptx
		OpenVDB::nanovdb
		imgui::imgui
)
