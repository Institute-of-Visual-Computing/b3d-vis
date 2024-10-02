
embed_ptx(
  OUTPUT_TARGET
    "FastVoxelTraversal-ptx"
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    FastVoxelTraversalDeviceCode.cu
)

set_property(TARGET FastVoxelTraversal-ptx PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET FastVoxelTraversal-ptx_ptx PROPERTY FOLDER "Renderer/PTX Targets")

find_package(tinyxml2 CONFIG REQUIRED)
find_package(unofficial-cfitsio CONFIG REQUIRED)

find_package(Stb REQUIRED)
find_package(imgui CONFIG REQUIRED)

add_b3d_renderer(
	NAME
		FastVoxelTraversal
	SOURCES
		FastVoxelTraversalRenderer.cpp SourceVolumeLoader.cpp
	HEADERS
		FastVoxelTraversalSharedStructs.h
	LIBRARIES
		"FastVoxelTraversal-ptx"
		imgui::imgui
        tinyxml2::tinyxml2
        cfitsio
)

target_include_directories(FastVoxelTraversal
	PRIVATE
		${Stb_INCLUDE_DIR}
)
