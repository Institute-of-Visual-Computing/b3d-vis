#include "Texture.h"

#include <cassert>
#include <cstring>

using namespace b3d::unity_cuda_interop;

auto Texture::unregisterCUDA() -> void
{
	assert(cudaGraphicsResource_ != nullptr);
	cudaGraphicsUnregisterResource(cudaGraphicsResource_);
	cudaGraphicsResource_ = nullptr;
}
