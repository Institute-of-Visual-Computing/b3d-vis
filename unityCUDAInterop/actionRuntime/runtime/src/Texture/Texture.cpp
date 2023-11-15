#include "Texture.h"

#include <cassert>
#include <cstring>

void Texture::unregisterCUDA()
{
	assert(cudaGraphicsResource_ != nullptr);
	cudaGraphicsUnregisterResource(cudaGraphicsResource_);
	cudaGraphicsResource_ = nullptr;
}
