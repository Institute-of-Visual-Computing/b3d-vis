#pragma once

#include <owl/common.h>

#ifndef __CUDACC__
template <typename T>
auto tex2D(cudaTextureObject_t, float, float) -> T
{
	return {};
}

auto surf2Dwrite(uint32_t, cudaSurfaceObject_t, size_t, int) -> void {}
#endif
