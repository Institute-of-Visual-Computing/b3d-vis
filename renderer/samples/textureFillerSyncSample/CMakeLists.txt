add_b3d_renderer(
	NAME
		TextureFillerSyncSample
	SOURCES
		TextureFillerSyncSample.cu
	HEADERS
		TextureFillerSyncSample.h
)

set_target_properties(TextureFillerSyncSample PROPERTIES CUDA_ARCHITECTURES "72")


