embed_ptx(
	OUTPUT_TARGET
		nano-ptx
	PTX_LINK_LIBRARIES
		owl::owl
		B3D::Renderer::Framework
	SOURCES
		NanoRenderer.cu
		SampleAccumulators.h
		SamplerMapper.h
	#DEVICE_DEBUG
)


set_property(TARGET nano-ptx PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET nano-ptx_ptx PROPERTY FOLDER "Renderer/PTX Targets")

embed_optixir(
	OUTPUT_TARGET
		nano-optixir
	OPTIXIR_LINK_LIBRARIES
		owl::owl	
		B3D::Renderer::Framework
	SOURCES
		NanoRenderer.cu
		SampleAccumulators.h
		SamplerMapper.h
	#DEVICE_DEBUG
)


set_property(TARGET nano-optixir PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET nano-optixir_optixir PROPERTY FOLDER "Renderer/PTX Targets")

set_source_files_properties(
	NanoRenderer.cu
	PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
 )

find_package(OpenVDB CONFIG REQUIRED COMPONENTS nanovdb)
find_package(imgui CONFIG REQUIRED)

# set_source_files_properties(
#	 FoveatedRendering.cu
#	 PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
# )


add_library(cudaCode FoveatedRendering.cu FoveatedRendering.h FoveatedHelper.cuh)
target_compile_features(cudaCode PUBLIC cxx_std_17)
target_include_directories(cudaCode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET cudaCode PROPERTY FOLDER "Renderer/Nano")

target_link_libraries(cudaCode
	PRIVATE
		B3D::Renderer::Framework
)
set_property(TARGET cudaCode PROPERTY CUDA_STANDARD 17)

find_package(Tracy CONFIG REQUIRED)

add_b3d_renderer(
	NAME
		Nano
	SOURCES
		NanoRenderer.cpp
		Old_OpenFileDialog.cpp
	HEADERS
		NanoRenderer.h
		Old_OpenFileDialog.h
	LIBRARIES
		cudaCode
		nano-ptx
		nano-optixir
		OpenVDB::nanovdb
		imgui::imgui
		B3D::Tools::ColorMap
		Tracy::TracyClient
		B3D::Header::Shared::NvdbRendering
		B3D::Tools::RuntimeDataset

)
