#include <device_launch_parameters.h>
#include <optix_device.h>
#include <owl/owl.h>

#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>
#include "SharedStructs.h"
#include "nanovdb/NanoVDB.h"
#include "owl/common/math/vec.h"

#include "owl/owl_device.h"

#include <array>

using namespace b3d::renderer::nano;
using namespace owl;

extern "C" __constant__ LaunchParams optixLaunchParams;

struct PerRayData
{
	vec3f color;
	float alpha;
	bool isBackground{ false };
};

inline __device__ void confine(const nanovdb::BBox<nanovdb::Coord>& bbox, nanovdb::Vec3f& sample)
{
	// NanoVDB's voxels and tiles are formed from half-open intervals, i.e.
	// voxel[0, 0, 0] spans the set [0, 1) x [0, 1) x [0, 1). To find a point's voxel,
	// its coordinates are simply truncated to integer. Ray-box intersections yield
	// pairs of points that, because of numerical errors, fall randomly on either side
	// of the voxel boundaries.
	// This confine method, given a point and a (integer-based/Coord-based) bounding
	// box, moves points outside the bbox into it. That means coordinates at lower
	// boundaries are snapped to the integer boundary, and in case of the point being
	// close to an upper boundary, it is move one EPS below that bound and into the volume.

	// get the tighter box around active values
	auto iMin = nanovdb::Vec3f(bbox.min());
	auto iMax = nanovdb::Vec3f(bbox.max()) + nanovdb::Vec3f(1.0f);

	// move the start and end points into the bbox
	float eps = 1e-7f;
	if (sample[0] < iMin[0])
		sample[0] = iMin[0];
	if (sample[1] < iMin[1])
		sample[1] = iMin[1];
	if (sample[2] < iMin[2])
		sample[2] = iMin[2];
	if (sample[0] >= iMax[0])
		sample[0] = iMax[0] - fmaxf(1.0f, fabsf(sample[0])) * eps;
	if (sample[1] >= iMax[1])
		sample[1] = iMax[1] - fmaxf(1.0f, fabsf(sample[1])) * eps;
	if (sample[2] >= iMax[2])
		sample[2] = iMax[2] - fmaxf(1.0f, fabsf(sample[2])) * eps;
}

inline __hostdev__ void confine(const nanovdb::BBox<nanovdb::Coord>& bbox, nanovdb::Vec3f& start, nanovdb::Vec3f& end)
{
	confine(bbox, start);
	confine(bbox, end);
}


OPTIX_BOUNDS_PROGRAM(volumeBounds)
(const void* geometryData, owl::box3f& primitiveBounds, const int primitiveID)
{
	const auto& self = *static_cast<const GeometryData*>(geometryData);
	primitiveBounds = self.volume.indexBox;
}

OPTIX_RAYGEN_PROGRAM(hitCountRayGen)()
{
}

OPTIX_RAYGEN_PROGRAM(rayGeneration)()
{
	const auto& self = owl::getProgramData<RayGenerationData>();


	const auto& camera = optixLaunchParams.cameraData;
	const auto pixelId = owl::getLaunchIndex();

	const auto screen = (vec2f(pixelId) + vec2f(.5f)) / vec2f(self.frameBufferSize); //*2.0f -1.0f;

	owl::Ray ray;
	ray.origin = camera.pos;
	ray.direction = normalize(camera.dir_00 + screen.x * camera.dir_du + screen.y * camera.dir_dv);

	PerRayData prd;
	owl::traceRay(self.world, ray, prd);

	const auto bg1 = optixLaunchParams.bg.color1;
	const auto bg2 = optixLaunchParams.bg.color0;
	const auto pattern = (pixelId.x / 8) ^ (pixelId.y / 8);
	auto bgColor = (pattern & 1) ? bg1 : bg2;
	const auto a = prd.alpha;
	if (optixLaunchParams.bg.fillBox && !prd.isBackground)
	{

		bgColor = optixLaunchParams.bg.fillColor;
	}

	const auto color = prd.isBackground ? bgColor : optixLaunchParams.color * (1 - a) + a * bgColor;

	surf2Dwrite(owl::make_rgba(color), optixLaunchParams.surfacePointer, sizeof(uint32_t) * pixelId.x, pixelId.y);
}

OPTIX_MISS_PROGRAM(miss)()
{
	auto& prd = owl::getPRD<PerRayData>();
	prd.isBackground = true;
}

OPTIX_CLOSEST_HIT_PROGRAM(nano_closestHit)()
{
	const auto& geometry = owl::getProgramData<GeometryData>();
	const auto* grid = reinterpret_cast<nanovdb::FloatGrid*>(geometry.volume.grid);

	float transform[12];
	optixGetWorldToObjectTransformMatrix(transform);

	float indexToWorldTransform[9];
	indexToWorldTransform[0] = transform[0];
	indexToWorldTransform[1] = transform[1];
	indexToWorldTransform[2] = transform[2];
	indexToWorldTransform[3] = transform[4];
	indexToWorldTransform[4] = transform[5];
	indexToWorldTransform[5] = transform[6];
	indexToWorldTransform[6] = transform[8];
	indexToWorldTransform[7] = transform[9];
	indexToWorldTransform[8] = transform[10];
	const auto translate = nanovdb::Vec3f{ transform[3], transform[7], transform[11] };


	const auto& accessor = grid->getAccessor();

	const auto rayOrigin = optixGetWorldRayOrigin();
	const auto rayDirection = optixGetWorldRayDirection();

	const auto t0 = optixGetRayTmax();
	const auto t1 = getPRD<float>();

	const auto rayWorld = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(rayOrigin),
											  reinterpret_cast<const nanovdb::Vec3f&>(rayDirection));

	const auto startWorld = rayWorld(t0);
	const auto endWorld = rayWorld(t1);
	const auto a = nanovdb::Vec3f(startWorld[0], startWorld[1], startWorld[2]);
	const auto b = nanovdb::Vec3f(endWorld[0], endWorld[1], endWorld[2]);
	auto start = nanovdb::matMult(&indexToWorldTransform[0], a) + translate;
	auto end = nanovdb::matMult(&indexToWorldTransform[0], b) + translate;





	const auto bbox = grid->indexBBox();
	confine(bbox, start, end);

	const auto direction = end - start;
	const auto length = direction.length();
	const auto ray = nanovdb::Ray<float>(start, direction / length, 0.0f, length);
	auto ijk = nanovdb::RoundDown<nanovdb::Coord>(ray.start());

	auto hdda = nanovdb::HDDA<nanovdb::Ray<float>>(ray, accessor.getDim(ijk, ray));

	const auto opacity = 10.0f; // 0.01f;//1.0.f;
	auto transmittance = 1.0f;
	auto t = 0.0f;
	auto density = accessor.getValue(ijk) * opacity;
	while (hdda.step())
	{
		const auto dt = hdda.time() - t;
		transmittance *= expf(-density * dt);
		t = hdda.time();
		ijk = hdda.voxel();
		const auto value = accessor.getValue(ijk);
		density = value * opacity;
		hdda.update(ray, accessor.getDim(ijk, ray));
	}

	auto& prd = owl::getPRD<PerRayData>();

	prd.color = vec3f(0.8, 0.3, 0.2) * transmittance;
	prd.alpha = transmittance;
}

OPTIX_INTERSECT_PROGRAM(nano_intersection)()
{
	const auto& geometry = owl::getProgramData<GeometryData>();
	const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(geometry.volume.grid);

	const auto rayOrigin = optixGetObjectRayOrigin();
	const auto rayDirection = optixGetObjectRayDirection();

	const auto& bbox = grid->worldBBox();
	auto t0 = optixGetRayTmin();
	auto t1 = optixGetRayTmax();
	const auto ray = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(rayOrigin),
										 reinterpret_cast<const nanovdb::Vec3f&>(rayDirection), t0, t1);

	if (ray.intersects(bbox, t0, t1))
	{
		auto& t = getPRD<float>();
		t = t1;

		optixReportIntersection(fmaxf(t0, optixGetRayTmin()), 0);
	}
}
