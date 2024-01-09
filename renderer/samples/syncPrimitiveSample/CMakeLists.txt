add_b3d_renderer(
	NAME
		SyncPrimitiveSample
	SOURCES
		SyncPrimitiveSampleRenderer.cu
	HEADERS
		SyncPrimitiveSampleRenderer.h
)

set_target_properties(SyncPrimitiveSample PROPERTIES CUDA_ARCHITECTURES "72")
