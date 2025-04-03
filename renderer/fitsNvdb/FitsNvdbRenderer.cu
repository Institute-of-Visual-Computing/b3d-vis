#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <optix_device.h>

#include <owl/owl.h>

#include "Common.h"
#include "SharedStructs.h"

#include <owl/owl_device.h>
#include "FitsNvdbRenderer.h"
#include "OptixHelper.cuh"

#include <device_launch_parameters.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/Ray.h>
#include "SampleAccumulators.h"

using namespace owl;
using namespace b3d::renderer::fitsNvdb;

extern "C" __constant__ LaunchParams optixLaunchParams;

struct PerRayData
{
	bool middlePos{ false };
	float tFar;
	vec3f color;
	float alpha;
	bool isBackground{ false };
	float stepsScale{ 1.0 };
};

__device__ inline auto confine(const nanovdb::math::BBox<nanovdb::Coord>& bbox, nanovdb::Vec3f& sample) -> void
{

	auto iMin = nanovdb::Vec3f(bbox.min());
	auto iMax = nanovdb::Vec3f(bbox.max()) + nanovdb::Vec3f(1.0f);

	constexpr auto eps = 1e-7f;
	if (sample[0] < iMin[0])
	{
		sample[0] = iMin[0];
	}
	if (sample[1] < iMin[1])
	{
		sample[1] = iMin[1];
	}
	if (sample[2] < iMin[2])
	{
		sample[2] = iMin[2];
	}
	if (sample[0] >= iMax[0])
	{
		sample[0] = iMax[0] - fmaxf(1.0f, fabsf(sample[0])) * eps;
	}
	if (sample[1] >= iMax[1])
	{
		sample[1] = iMax[1] - fmaxf(1.0f, fabsf(sample[1])) * eps;
	}
	if (sample[2] >= iMax[2])
	{
		sample[2] = iMax[2] - fmaxf(1.0f, fabsf(sample[2])) * eps;
	}
}

__device__ inline auto confine(const nanovdb::math::BBox<nanovdb::Coord>& bbox, nanovdb::Vec3f& start,
							   nanovdb::Vec3f& end) -> void
{
	confine(bbox, start);
	confine(bbox, end);
}

OPTIX_BOUNDS_PROGRAM(bounds)
(const void* geometryData, owl::box3f& primitiveBounds, const int primitiveID)
{
	const FitsNvdbGeometry& self = *(const FitsNvdbGeometry*)geometryData;

	// printf("Min: %.1f, %.1f, %.1f", self.fitsBox.lower.x, self.fitsBox.lower.y, self.fitsBox.lower.z);
	// printf("max: %.1f, %.1f, %.1f", self.fitsBox.upper.x, self.fitsBox.upper.y, self.fitsBox.upper.z);
	primitiveBounds = self.fitsBox;
}

__host__ __device__ auto b3d::renderer::nano::transferMap(const float a) -> float
{
	return tex2D<float>(optixLaunchParams.transferFunctionTexture, a, 0.5f);
}

__host__ __device__ auto b3d::renderer::nano::colorMap(const float value) -> owl::vec3f
{
	if (optixLaunchParams.coloringInfo.coloringMode == ColoringMode::single)
	{
		return owl::vec3f{ optixLaunchParams.coloringInfo.singleColor };
	}
	else
	{
		const auto result =
			tex2D<float4>(optixLaunchParams.colorMaps, value, optixLaunchParams.coloringInfo.selectedColorMap);
		return owl::vec3f{ result.x, result.y, result.z };
	}
}

OPTIX_RAYGEN_PROGRAM(raygen)()
{
	const auto& self = owl::getProgramData<RayGenerationData>();

	const auto& camera = optixLaunchParams.cameraData;
	const auto pixelId = owl::getLaunchIndex();

	const auto screen = (vec2f(pixelId) + vec2f(.5f)) / vec2f(self.frameBufferSize); //*2.0f -1.0f;

	owl::Ray ray;
	ray.origin = camera.pos;
	ray.direction = normalize(camera.dir_00 + screen.x * camera.dir_du + screen.y * camera.dir_dv);

	PerRayData prd;
	prd.middlePos = pixelId.x == self.frameBufferSize.x / 2 && pixelId.y == self.frameBufferSize.y / 2;
	owl::traceRay(self.world, ray, prd);
	const auto bg1 = optixLaunchParams.bg.color1;
	const auto bg2 = optixLaunchParams.bg.color0;
	const auto pattern = (pixelId.x / 8) ^ (pixelId.y / 8);

	const auto bgColor = (pattern & 1) ? bg1 : bg2;

	const auto color = prd.isBackground ? bgColor : vec4f(prd.color, prd.alpha);


	if (pixelId.x == self.frameBufferSize.x / 2 && pixelId.y == self.frameBufferSize.y / 2)
	{
		// printf("Hello from the center %d\n", true);
	}

	surf2Dwrite(owl::make_rgba(color), optixLaunchParams.surfacePointer, sizeof(uint32_t) * pixelId.x, pixelId.y);
}

OPTIX_MISS_PROGRAM(miss)()
{
	auto& prd = owl::getPRD<PerRayData>();
	prd.isBackground = true;
}

OPTIX_INTERSECT_PROGRAM(intersect)()
{
	const nanovdb::FloatGrid* grid = reinterpret_cast<nanovdb::FloatGrid*>(optixLaunchParams.volume.grid);

	const auto rayOrigin = optixGetObjectRayOrigin();
	const auto rayDirection = optixGetObjectRayDirection();

	auto& prd = getPRD<PerRayData>();

	nanovdb::Vec3d min = { 1.0, 1.0, 1.0 };
	min = { 0.0, 0.0, 0.0 };
	nanovdb::Vec3d max = { 3, 3, 3 };
	// const auto bbox = nanovdb::BBox<nanovdb::Vec3d>(min, max);

	const auto& bbox = grid->worldBBox();

	auto t0 = optixGetRayTmin();
	auto t1 = optixGetRayTmax();
	const auto ray = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(rayOrigin),
										 reinterpret_cast<const nanovdb::Vec3f&>(rayDirection), t0, t1);

	if (ray.intersects(bbox, t0, t1))
	{
		// auto& t = getPRD<float>();
		// t = t1;
		optixReportIntersection(fmaxf(t0, optixGetRayTmin()), 0);
		optixReportIntersection(fmaxf(t0, optixGetRayTmin()), 0, __float_as_int(t1));
	}
}

OPTIX_CLOSEST_HIT_PROGRAM(closestHit)()
{

	// const auto t1 = getPRD<float>();
	const nanovdb::FloatGrid* grid = reinterpret_cast<nanovdb::FloatGrid*>(optixLaunchParams.volume.grid);
	const auto& accessor = grid->getAccessor();

	auto transform = cuda::std::array<float, 12>{};
	optixGetWorldToObjectTransformMatrix(transform.data());
	auto indexToWorldTransform =
		cuda::std::array<float, 9>{ { transform[0], transform[1], transform[2], transform[4], transform[5],
									  transform[6], transform[8], transform[9], transform[10] } };
	const auto translate = nanovdb::Vec3f{ transform[3], transform[7], transform[11] };


	const auto rayOrigin = optixGetWorldRayOrigin();
	const auto rayDirection = optixGetWorldRayDirection();
	const auto rayWorld = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(rayOrigin),
											  reinterpret_cast<const nanovdb::Vec3f&>(rayDirection));

	const auto t0 = optixGetRayTmax();
	const auto t1 = __int_as_float(optixGetAttribute_0());
	const auto startWorld = rayWorld(t0);
	const auto endWorld = rayWorld(t1);

	const auto a = nanovdb::Vec3f(startWorld[0], startWorld[1], startWorld[2]);
	const auto b = nanovdb::Vec3f(endWorld[0], endWorld[1], endWorld[2]);
	auto start = nanovdb::math::matMult(indexToWorldTransform.data(), a) + translate;
	auto end = nanovdb::math::matMult(indexToWorldTransform.data(), b) + translate;

	const auto& bbox = grid->indexBBox();
	confine(bbox, start, end);

	const auto direction = end - start;
	const auto length = direction.length();
	const auto ray = nanovdb::math::Ray<float>(start, direction / length, 0.0f, length);
	auto ijk = nanovdb::math::RoundDown<nanovdb::Coord>(ray.start());

	auto hdda = nanovdb::math::HDDA<nanovdb::math::Ray<float>>(ray, accessor.getDim(ijk, ray));

	auto result = vec4f{};
	auto& prd = owl::getPRD<PerRayData>();

	auto remapSample = [](const float value)
	{
		return optixLaunchParams.sampleRemapping.x +
			value / (optixLaunchParams.sampleRemapping.y - optixLaunchParams.sampleRemapping.x);
	};

	const auto integrate = [&](auto sampleAccumulator)
	{
		sampleAccumulator.preAccumulate();

		const auto acc = grid->tree().getAccessor();

		while (hdda.step())
		{
			// TODO: can we remove step and update? Investigate examples in optix sdk
			const auto tt = hdda.next();
			ijk = hdda.voxel();
			const auto value = remapSample(accessor.getValue(nanovdb::Coord::Floor(ray(tt))));
			sampleAccumulator.accumulate(value);
			hdda.update(ray, accessor.getDim(ijk, ray));
		}

		sampleAccumulator.postAccumulate();
		return sampleAccumulator.getAccumulator();
	};

	switch (optixLaunchParams.sampleIntegrationMethod)
	{

	case SampleIntegrationMethod::transferIntegration:
		{
			auto sampleAccumulator = b3d::renderer::nano::IntensityIntegration{};
			result = integrate(sampleAccumulator);
		}

		break;
	case SampleIntegrationMethod::maximumIntensityProjection:
		{
			auto sampleAccumulator = b3d::renderer::nano::MaximumIntensityProjection{};
			result = integrate(sampleAccumulator);
		}

		break;
	case SampleIntegrationMethod::averageIntensityProjection:
		{
			auto sampleAccumulator = b3d::renderer::nano::AverageIntensityProjection{};
			result = integrate(sampleAccumulator);
		}
		break;
	}

	prd.color = vec3f{ result.x, result.y, result.z };
	prd.alpha = result.w;
}
