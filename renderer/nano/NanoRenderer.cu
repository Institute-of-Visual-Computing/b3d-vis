#include <device_launch_parameters.h>
#include <optix_device.h>
#include <owl/owl.h>

#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>
#include "SharedStructs.h"
#include "nanovdb/NanoVDB.h"
#include "owl/common/math/vec.h"

#include "owl/owl_device.h"

using namespace b3d::renderer::nano;
using namespace owl;

extern "C" __constant__ LaunchParams optixLaunchParams;


struct PerRayData
{
	vec3f color;
};

inline __device__ void confine(const nanovdb::BBox<nanovdb::Coord>& bbox, nanovdb::Vec3f& iVec)
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
	if (iVec[0] < iMin[0])
		iVec[0] = iMin[0];
	if (iVec[1] < iMin[1])
		iVec[1] = iMin[1];
	if (iVec[2] < iMin[2])
		iVec[2] = iMin[2];
	if (iVec[0] >= iMax[0])
		iVec[0] = iMax[0] - fmaxf(1.0f, fabsf(iVec[0])) * eps;
	if (iVec[1] >= iMax[1])
		iVec[1] = iMax[1] - fmaxf(1.0f, fabsf(iVec[1])) * eps;
	if (iVec[2] >= iMax[2])
		iVec[2] = iMax[2] - fmaxf(1.0f, fabsf(iVec[2])) * eps;
}

inline __hostdev__ void confine(const nanovdb::BBox<nanovdb::Coord>& bbox, nanovdb::Vec3f& iStart, nanovdb::Vec3f& iEnd)
{
	confine(bbox, iStart);
	confine(bbox, iEnd);
}


OPTIX_BOUNDS_PROGRAM(volumeBounds)
(const void* geometryData, owl::box3f& primitiveBounds, const int primitiveID)
{
	const auto& self = *static_cast<const GeometryData*>(geometryData);
	primitiveBounds = self.volume.worldAabb;
}

OPTIX_RAYGEN_PROGRAM(rayGeneration)()
{
	const auto& self = owl::getProgramData<RayGenerationData>();

	const int eyeIdx = optixLaunchParams.outputSurfaceIndex;
	const auto& camera = self.camera;
	const auto pixelId = owl::getLaunchIndex();

	const auto screen = (vec2f(pixelId) + vec2f(.5f)) / vec2f(self.frameBufferSize);//*2.0f -1.0f;

	owl::Ray ray;
	ray.origin = camera.position;
	ray.direction = normalize(camera.dir00 + screen.x * camera.dirDu + screen.y * camera.dirDv);

	PerRayData prd;
	owl::traceRay(self.world, ray, prd);

	const auto color = prd.color;
	surf2Dwrite(owl::make_rgba(color), self.frameBufferPtr[0],
				sizeof(uint32_t) * pixelId.x, pixelId.y);
}

OPTIX_MISS_PROGRAM(miss)()
{
	const auto pixelId = owl::getLaunchIndex();

	const auto& self = owl::getProgramData<MissProgramData>();

	auto& prd = owl::getPRD<PerRayData>();
	const auto pattern = (pixelId.x / 8) ^ (pixelId.y / 8);
	prd.color = (pattern & 1) ? self.color1 : self.color0;
}

OPTIX_CLOSEST_HIT_PROGRAM(nano_closesthit)()
{
	{
		auto& prd = owl::getPRD<PerRayData>();
		prd.color = vec3f(0.8,0.3,0.2);
		return;
	}
	
	const auto& geometry = owl::getProgramData<GeometryData>();
	const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(geometry.volume.grid);

	const auto& tree = grid->tree();
	const auto& accessor = tree.getAccessor();


	const auto rayOrigin = optixGetWorldRayOrigin();
	const auto rayDirection = optixGetWorldRayDirection();

	const auto t0 = optixGetRayTmin();
	const auto t1 = optixGetRayTmax();

	const auto rayWorld = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(rayOrigin),
											  reinterpret_cast<const nanovdb::Vec3f&>(rayDirection));
	auto start = grid->worldToIndexF(rayWorld(t0));
	auto end = grid->worldToIndexF(rayWorld(t1));

	const auto bbox = grid->indexBBox();
	confine(bbox, start, end);


	const auto direction = end - start;
	const auto length = direction.length();
	const auto ray = nanovdb::Ray<float>(start, direction / length, 0.0f, length);
	auto ijk = nanovdb::RoundDown<nanovdb::Coord>(ray.start());


	auto hdda = nanovdb::HDDA<nanovdb::Ray<float>>(ray, accessor.getDim(ijk, ray));

	const auto opacity = 1.0f;
	auto transmittance = 1.0f;
	auto t = 0.0f;
	auto density = accessor.getValue(ijk) * opacity;
	while (hdda.step())
	{
		const auto dt = hdda.time() - t;
		transmittance *= expf(-density * dt);
		t = hdda.time();
		ijk = hdda.voxel();

		density = accessor.getValue(ijk) * opacity;
		hdda.update(ray, accessor.getDim(ijk, ray));
	}

	auto& prd = owl::getPRD<PerRayData>();

	prd.color = vec3f(0.8,0.3,0.2) * transmittance;
}

OPTIX_INTERSECT_PROGRAM(nano_intersection)()
{
	const auto& geometry = owl::getProgramData<GeometryData>();
	const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(geometry.volume.grid);

	const auto rayOrigin = optixGetObjectRayOrigin();
	const auto rayDirection = optixGetObjectRayDirection();

	const auto bbox = grid->indexBBox();
	auto t0 = optixGetRayTmin();
	auto t1 = optixGetRayTmax();
	const auto ray = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(rayOrigin),
										 reinterpret_cast<const nanovdb::Vec3f&>(rayDirection), t0, t1);


	if (ray.intersects(bbox, t0, t1))
	{
		optixReportIntersection(fmaxf(t0, optixGetRayTmin()), 0);
	}
}
