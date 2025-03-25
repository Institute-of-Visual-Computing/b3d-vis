#pragma once

#include <owl/common.h>

#ifndef __CUDACC__
template <typename T>
auto tex2D(cudaTextureObject_t, float, float) -> T
{
	return {};
}
auto surf2Dwrite(uint32_t, cudaSurfaceObject_t, size_t, int) -> void
{
}
static __forceinline__ __device__ unsigned int optixGetPrimitiveIndex()
{
	return {};
}
static __forceinline__ __device__ int __float_as_int(float value)
{
	return int{};
}
static __forceinline__ __device__ float __int_as_float(int value)
{
	return float{};
}
static __forceinline__ __device__ float3 optixGetWorldRayOrigin()
{
	return float3{};
}
static __forceinline__ __device__ float3 optixGetObjectRayOrigin()
{
	return float3{};
}
static __forceinline__ __device__ float3 optixGetWorldRayDirection()
{
	return float3{};
}
static __forceinline__ __device__ float3 optixGetObjectRayDirection()
{
	return float3{};
}
static __forceinline__ __device__ float2 optixGetTriangleBarycentrics()
{
	return float2{};
}
static __forceinline__ __device__ void optixGetWorldToObjectTransformMatrix(float m[12])
{
}
static __forceinline__ __device__ float optixGetRayTmin()
{
	return float{};
}
static __forceinline__ __device__ float optixGetRayTmax()
{
	return float{};
}
static __forceinline__ __device__ bool optixReportIntersection(float hitT, unsigned int hitKind)
{
	return bool{};
}
static __forceinline__ __device__ bool optixReportIntersection(float hitT, unsigned int hitKind, unsigned int a0)
{
	return bool{};
}

static __forceinline__ __device__ bool optixReportIntersection(float hitT, unsigned int hitKind, unsigned int a0,
															   unsigned int a1)
{
	return bool{};
}

static __forceinline__ __device__ bool optixReportIntersection(float hitT, unsigned int hitKind, unsigned int a0,
															   unsigned int a1, unsigned int a2)
{
	return bool{};
}

static __forceinline__ __device__ bool optixReportIntersection(float hitT, unsigned int hitKind, unsigned int a0,
															   unsigned int a1, unsigned int a2, unsigned int a3)
{
	return bool{};
}

static __forceinline__ __device__ bool optixReportIntersection(float hitT, unsigned int hitKind, unsigned int a0,
															   unsigned int a1, unsigned int a2, unsigned int a3,
															   unsigned int a4)
{
	return bool{};
}

static __forceinline__ __device__ bool optixReportIntersection(float hitT, unsigned int hitKind, unsigned int a0,
															   unsigned int a1, unsigned int a2, unsigned int a3,
															   unsigned int a4, unsigned int a5)
{
	return bool{};
}

static __forceinline__ __device__ bool optixReportIntersection(float hitT, unsigned int hitKind, unsigned int a0,
															   unsigned int a1, unsigned int a2, unsigned int a3,
															   unsigned int a4, unsigned int a5, unsigned int a6)
{
	return bool{};
}

static __forceinline__ __device__ bool optixReportIntersection(float hitT, unsigned int hitKind, unsigned int a0,
															   unsigned int a1, unsigned int a2, unsigned int a3,
															   unsigned int a4, unsigned int a5, unsigned int a6,
															   unsigned int a7)
{
	return bool{};
}
static __forceinline__ __device__ unsigned int optixGetAttribute_0()
{
	return unsigned int{};
}
static __forceinline__ __device__ unsigned int optixGetAttribute_1()
{
	return unsigned int{};
}
static __forceinline__ __device__ unsigned int optixGetAttribute_2()
{
	return unsigned int{};
}
static __forceinline__ __device__ unsigned int optixGetAttribute_3()
{
	return unsigned int{};
}
static __forceinline__ __device__ unsigned int optixGetAttribute_4()
{
	return unsigned int{};
}
static __forceinline__ __device__ unsigned int optixGetAttribute_5()
{
	return unsigned int{};
}
static __forceinline__ __device__ unsigned int optixGetAttribute_6()
{
	return unsigned int{};
}
static __forceinline__ __device__ unsigned int optixGetAttribute_7()
{
	return unsigned int{};
}
#endif
