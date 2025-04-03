find_package(Tracy CONFIG REQUIRED)

embed_optixir(
	OUTPUT_TARGET
		fitsNvdb-optixir
	OPTIXIR_LINK_LIBRARIES
		owl::owl	
		B3D::Renderer::Framework
	SOURCES
		FitsNvdbRenderer.cu
		SampleAccumulators.h
		SamplerMapper.h
	#DEVICE_DEBUG
)
target_compile_options(fitsNvdb-optixir_optixir PRIVATE -rdc=true)
target_compile_options(fitsNvdb-optixir_optixir PRIVATE -diag-suppress 1427)


set_property(TARGET fitsNvdb-optixir PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET fitsNvdb-optixir_optixir PROPERTY FOLDER "Renderer/PTX Targets")

embed_ptx(
	OUTPUT_TARGET
		fitsNvdb-ptx
	PTX_LINK_LIBRARIES
		owl::owl
		B3D::Renderer::Framework
	SOURCES
		FitsNvdbRenderer.cu
		SampleAccumulators.h
		SamplerMapper.h
	#DEVICE_DEBUG
)

target_compile_options(fitsNvdb-ptx_ptx PRIVATE -rdc=true)
target_compile_options(fitsNvdb-ptx_ptx PRIVATE -diag-suppress 1427)

set_property(TARGET fitsNvdb-ptx PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET fitsNvdb-ptx_ptx PROPERTY FOLDER "Renderer/PTX Targets")

set_source_files_properties(
	FitsNvdbRenderer.cu
	PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
 )

add_b3d_renderer(
	NAME
		FitsNvdb
	SOURCES
		FitsNvdbRenderer.cpp
	HEADERS
		FitsNvdbRenderer.h
		SharedStructs.h
	LIBRARIES
		fitsNvdb-optixir
		fitsNvdb-ptx
		Tracy::TracyClient
		B3D::Header::Shared::NvdbRendering
)
