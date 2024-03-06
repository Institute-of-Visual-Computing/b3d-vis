#pragma once

#include <owl/common/math/vec.h>


namespace b3d
{
	namespace renderer
	{
		namespace nano
		{

			__host__ __device__ inline auto colorMap(float value) -> owl::vec3f;
			__host__ __device__ inline auto transferMap(float a) -> float;
		}
	}
}
