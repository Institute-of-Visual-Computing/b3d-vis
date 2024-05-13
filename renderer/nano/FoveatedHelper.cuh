#pragma once
#include <crt/math_functions.h>
#include <owl/common.h>

#include "cuda_runtime.h"
#include "math_constants.h"

template<typename T>
inline __device__ auto length(const owl::vec_t<T, 2>& v) -> T
{
	return owl::common::polymorphic::sqrt(dot(v, v));
}

inline __device__ auto maximumLogDistance(const owl::vec2f& foveal, const owl::vec2f& resolution) -> float
{
	const auto maxL = max(
		max(
			length((owl::vec2f(1, 1) - foveal) * resolution),
			length((owl::vec2f(1, -1) - foveal) * resolution)),
		max(
			length((owl::vec2f(-1, 1) - foveal) * resolution),
			length((owl::vec2f(-1, -1) - foveal) * resolution)));
	const float L = log(maxL * 0.5);
	return L;
}

inline __device__ auto inverseLogMap(const float scaleRatio, const owl::vec2f& coordinateScreenSpace, const owl::vec2f& foveal,
	const owl::vec2f& resolution) -> owl::vec2f
{
	const auto L = maximumLogDistance(foveal, resolution);
	constexpr auto pi2 = CUDART_PI_F * 2.0f;


	const auto pq = coordinateScreenSpace / resolution * 2.0f - 1.0f - foveal;
	const auto lr = pow(log(length(pq * resolution * 0.5f)) / L, 4.0);
	const float theta = atan2f(pq.y * resolution.y, pq.x * resolution.x) / pi2 + (pq.y < 0.0f ? 1.0f : 0.0);

	const auto logCoordinate = owl::vec2f(lr, theta) / scaleRatio;
	return logCoordinate * resolution + foveal;
}

inline __device__ auto logMap(const float scaleRatio, const owl::vec2f& coordinate, const owl::vec2f& foveal, const owl::vec2f& resolution)
-> owl::vec2f
{
	const auto L = maximumLogDistance(foveal, resolution);
	auto uv = scaleRatio * coordinate / resolution;

	uv.x = pow(uv.x, 1.0f / 4.0f);
	constexpr auto pi2 = CUDART_PI_F  * 2.0f;
	const auto x = exp(uv.x * L) * cos(uv.y * pi2);
	const auto y = exp(uv.x * L) * sin(uv.y * pi2);
	const auto logCoordinate = owl::vec2f(x, y) + (foveal + owl::vec2f(1.f)) * 0.5f * resolution;

	return logCoordinate * scaleRatio;
}
