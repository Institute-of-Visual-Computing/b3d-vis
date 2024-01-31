embed_ptx(
	OUTPUT_TARGET
		nano-ptx
	PTX_LINK_LIBRARIES
		owl::owl
	SOURCES
		NanoRenderer.cu
	#DEVICE_DEBUG
)

embed_optixir(
	OUTPUT_TARGET
		nano-optixir
	OPTIXIR_LINK_LIBRARIES
		owl::owl
	SOURCES
		NanoRenderer.cu
	#DEVICE_DEBUG
)

set_source_files_properties(NanoRenderer.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
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
		nano-optixir
		OpenVDB::nanovdb
		imgui::imgui
		NanoCutterParser
)
