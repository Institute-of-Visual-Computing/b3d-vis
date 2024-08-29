embed_ptx(
	OUTPUT_TARGET
		nanoOutOfCore-ptx
	PTX_LINK_LIBRARIES
		owl::owl
		B3D::Renderer::Library
	SOURCES
		NanoOutOfCoreRenderer.cu

	#DEVICE_DEBUG
)


set_property(TARGET nanoOutOfCore-ptx PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET nanoOutOfCore-ptx_ptx PROPERTY FOLDER "Renderer/PTX Targets")

embed_optixir(
	OUTPUT_TARGET
		nanoOutOfCore-optixir
	OPTIXIR_LINK_LIBRARIES
		owl::owl
		B3D::Renderer::Library
	SOURCES
		NanoOutOfCoreRenderer.cu
	#DEVICE_DEBUG
)


set_property(TARGET nanoOutOfCore-optixir PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET nanoOutOfCore-optixir_optixir PROPERTY FOLDER "Renderer/PTX Targets")

set_source_files_properties(NanoRenderer.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
  )

find_package(OpenVDB CONFIG REQUIRED COMPONENTS nanovdb)

find_package(imgui CONFIG REQUIRED)

add_b3d_renderer(
	NAME
		NanoOutOfCore
	SOURCES
		NanoOutOfCoreRenderer.cpp
		Old_OpenFileDialog.cpp
	HEADERS
		NanoOutOfCoreRenderer.h
		Old_OpenFileDialog.h
	LIBRARIES
		nanoOutOfCore-ptx
		nanoOutOfCore-optixir
		OpenVDB::nanovdb
		imgui::imgui
		NanoCutterParser
)
