embed_ptx(
	OUTPUT_TARGET
		nano-ptx
	PTX_LINK_LIBRARIES
		owl::owl
		B3D::Renderer::Library
	SOURCES
		NanoRenderer.cu
	#DEVICE_DEBUG
)


set_property(TARGET nano-ptx PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET nano-ptx_ptx PROPERTY FOLDER "Renderer/PTX Targets")

embed_optixir(
	OUTPUT_TARGET
		nano-optixir
	OPTIXIR_LINK_LIBRARIES
		owl::owl	
		B3D::Renderer::Library
	SOURCES
		NanoRenderer.cu
	#DEVICE_DEBUG
)


set_property(TARGET nano-optixir PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET nano-optixir_optixir PROPERTY FOLDER "Renderer/PTX Targets")

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
		B3D::Tools::ColorMap
)
