// #include <math_constants.h>

#include <optix_device.h>

#include <device_launch_parameters.h>

#include "FastVoxelTraversalSharedStructs.h"
#include "owl/owl_device.h"

#include "cuda_runtime.h"
#include "owl/common/math/vec.h"

namespace
{
	inline __device__ vec3i sign(vec3f v)
	{
		return vec3i{ v.x < 0 ? -1 : v.x > 0 ? 1 : 0, v.y < 0 ? -1 : v.y > 0 ? 1 : 0, v.z < 0 ? -1 : v.z > 0 ? 1 : 0 };
	}

	inline __device__ vec3b lte(vec3f a, vec3f b)
	{

		{
			return { a.x <= b.x, a.y <= b.y, a.z <= b.z };
		}
	}

	inline __device__ vec3i ltei(vec3f a, vec3f b)
	{
		return { a.x <= b.x ? 1 : 0, a.y <= b.y ? 1 : 0, a.z <= b.z ? 1 : 0 };
	}

	__device__ vec3f mix(vec3f x, vec3f y, float s)
	{
		return x + s * (y - x);
	}
	inline __device__ vec3f spectral_jet(float x)
	{
		vec3f c;
		if (x < 0.25)
			c = vec3f(0.0, 4.0 * x, 1.0);
		else if (x < 0.5)
			c = vec3f(0.0, 1.0, 1.0 + 4.0 * (0.25 - x));
		else if (x < 0.75)
			c = vec3f(4.0 * (x - 0.5), 1.0, 0.0);
		else
			c = vec3f(1.0, 1.0 + 4.0 * (0.75 - x), 0.0);
		return clamp(c, vec3f(0.0), vec3f(1.0));
	}
} // namespace

extern "C" __constant__ MyLaunchParams optixLaunchParams;

__constant__ float EPS = .0001f;
#define PRINT_VEC3F(name, vec)                                                                                         \
	{                                                                                                                  \
		printf("%s: x: %.5f, y: %.5f, z: %.5f\n", name, (float)(vec).x, (float)(vec).y, (float)(vec).z);               \
	}

#define PRINT_VEC3I(name, vec)                                                                                         \
	{                                                                                                                  \
		printf("%s: x: %d, y: %d, z: %d\n", name, (int)(vec).x, (int)(vec).y, (int)(vec).z);               \
	}

#define PRINT_NEWLINE() printf("\n");
#define PRINT_HORIZ_LINE() printf("------------\n");

struct PerRayData
{
	vec4f color;
	vec2i fbSize;
	float hitCount;
	float maxVal;
	vec3f invDir;
	vec2i pxId;
};


OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
	const RayGenData& self = owl::getProgramData<RayGenData>();

	const RayCameraData& camData = optixLaunchParams.cameraData;

	const vec2i pixelID = owl::getLaunchIndex();

	const vec2f screen = (vec2f(pixelID) + vec2f(.5f)) / vec2f(self.fbSize);

	owl::Ray ray;
	ray.origin = optixLaunchParams.cameraData.pos;
	ray.direction = normalize(optixLaunchParams.cameraData.dir_00 + screen.u * optixLaunchParams.cameraData.dir_du +
							  screen.v * optixLaunchParams.cameraData.dir_dv);
	if (pixelID == self.fbSize / 2)
	{
		vec3f worldStartPos = camData.pos;
		// PRINT_VEC3F(worldStartPos);
	}
	
	PerRayData prd;
	prd.invDir = 1.0f / ray.direction;
	prd.pxId = pixelID;
	prd.fbSize = self.fbSize;
	prd.hitCount = 0;
	prd.maxVal = -CUDART_INF_F;
	owl::traceRay(/*accel to trace against*/ self.world,
				  /*the ray to trace*/ ray,
				  /*prd*/ prd);

	surf2Dwrite(owl::make_rgba(prd.color), optixLaunchParams.surfacePointer, sizeof(uint32_t) * pixelID.x, pixelID.y);
}

OPTIX_MISS_PROGRAM(miss)()
{
	const auto pixelId = owl::getLaunchIndex();

	const auto& self = owl::getProgramData<MissProgData>();

	auto& prd = owl::getPRD<PerRayData>();
	const auto pattern = (pixelId.x / 8) ^ (pixelId.y / 8);
	if (prd.hitCount > EPS && prd.color.w > EPS)
	{
		// keep this
	}
	else
	{
		prd.color = 0;
		//prd.color = (pattern & 1) ? self.color1 : self.color0;
	}
}


OPTIX_BOUNDS_PROGRAM(AABBGeom)(const void* geomData, box3f& primBounds, const int primID)
{
	const DatacubeSources& self = *(const DatacubeSources*)geomData;
	auto lower = self.sourceRegions[primID].sourceBoxNormalized.lower;
	auto upper = self.sourceRegions[primID].sourceBoxNormalized.upper;
	primBounds = { lower, upper };
}

OPTIX_INTERSECT_PROGRAM(AABBGeom)()
{
	const auto primID = optixGetPrimitiveIndex();
	const auto& datacubeSources = owl::getProgramData<DatacubeSources>();

	const vec3f rayOriginObj = optixGetObjectRayOrigin();
	vec3f rayDirectionObj = optixGetObjectRayDirection();
	const auto rayLengthInv = 1 / length(rayDirectionObj);
	rayDirectionObj = rayDirectionObj * rayLengthInv;

	auto& prd = owl::getPRD<PerRayData>();
	auto t_lower = (datacubeSources.sourceRegions[primID].sourceBoxNormalized.lower - rayOriginObj) / rayDirectionObj;
	auto t_upper = (datacubeSources.sourceRegions[primID].sourceBoxNormalized.upper - rayOriginObj) / rayDirectionObj;

	const auto tMin = min(t_lower, t_upper);
	const auto tMax = max(t_lower, t_upper);

	const auto tNear = max(max(tMin.x, tMin.y), tMin.z);
	const auto tFar = min(min(tMax.x, tMax.y), tMax.z);

	/*
	if (tNear < tFar && tFar > 0)
	{
	  optixReportIntersection(tNear * rayLengthInv, 0);
	}

	*/
	if (tFar >= tNear) // && !(tNear < 0 && tFar < 0))
	{
		float tFarReport = max(tNear, tFar);
		float tNearReport = max(0.0f, min(tNear, tFar));
		// Store both points for CH and AH program
		optixReportIntersection(tFar, 0, __float_as_int(tNear));
	}
}

OPTIX_CLOSEST_HIT_PROGRAM(AABBGeom)()
{
	/*
	auto& prd = owl::getPRD<PerRayData1>();

	const vec3f rayOrigin = optixGetWorldRayOrigin() ;
	const vec3f rayDir = optixGetWorldRayDirection();
	vec3f hitPosition = vec3f{ rayOrigin + optixGetRayTmax() * rayDir };
	prd.color = { optixTransformPointFromWorldToObjectSpace(hitPosition), 1.f };
	prd.color += vec4f{0.5f, .5f,.5f,.0f};
	*/
	// auto& prd = owl::getPRD<PerRayData>();
	// prd.color = { 1.0f, 0, 0, 1.0f };
}

OPTIX_ANY_HIT_PROGRAM(AABBGeom)()
{

	const DatacubeSources& self = owl::getProgramData<DatacubeSources>();
	PerRayData& prd = owl::getPRD<PerRayData>();

	float tFar = optixGetRayTmax();
	float tNear = __int_as_float(optixGetAttribute_0());
	auto primId = optixGetPrimitiveIndex();

	tNear = max(0.0f, min(tNear, tFar));
	vec3f rayDirection = optixGetWorldRayDirection();
	vec3f rayOrigin = optixGetWorldRayOrigin();

	vec3f worldPosEntry = rayOrigin + rayDirection * (tNear + EPS);
	vec3f worldPosExit = rayOrigin + rayDirection * (tFar + EPS);

	vec3f objPosEntry = optixTransformPointFromWorldToObjectSpace(worldPosEntry);
	vec3f objPosExit = optixTransformPointFromWorldToObjectSpace(worldPosExit);
		
	vec3f move = { 0, 0, 0 };
	move = { 0.5f, 0.5f, 0.5f };
	// Current position to position in grid coordinates
	vec3f worldGridPosEntryF = (objPosEntry + move) * vec3f{ self.gridDims };
	vec3f worldGridPosExitF = (objPosExit + move) * vec3f{ self.gridDims };

	// Change ray to local coordinates 0 to size of the box
	const vec3f localGridPosEntryF = worldGridPosEntryF - vec3f{ self.sourceRegions[primId].gridSourceBox.lower };
	const vec3f localGridPosExitF = worldGridPosExitF - vec3f{ self.sourceRegions[primId].gridSourceBox.lower };
	const vec3f localRayDirection = normalize(localGridPosExitF - localGridPosEntryF);

	// Box to test current position

	const box3i oaabb = box3i{ vec3i{ 0, 0, 0 }, self.sourceRegions[primId].gridSourceBox.size() };


	// The traversal algorithm consists of two phases : initialization and incremental traversal
	// The initialization phase begins by identifying the voxel in which the ray origin, ->u, is found.
	vec3i localGridPosI = vec3i{ localGridPosEntryF };


	// In addition, the variables stepX and stepY are initialized to either 1 or -1 indicating whether X and Y are
	// incremented or decremented as the ray crosses voxel boundaries(this is determined by the sign of the x and y
	// components of ->v).
	vec3i step = sign(localRayDirection);
	vec3f stepF = vec3f{ step };

	// TDeltaX indicates how far along the ray we must move
	// (in units of t) for the horizontal component of such a movement to equal the width of a voxel.Similarly,
	// we store in tDeltaY the amount of movement along the ray which has a vertical component equal to the
	// height of a voxel.
	vec3f tDelta = abs(vec3f(length(localRayDirection)) / localRayDirection);

	// Next, we determine the value of t at which the ray crosses the first vertical
	// voxel boundary and store it in variable tMaxX.We perform a similar computation
	// in y and store the result in tMaxY.
	// The minimum of these two values will indicate how much we can travel along the
	// ray and still remain in the current voxel.
	vec3f tMax = (stepF * (vec3f(localGridPosI) - localGridPosEntryF) + (stepF * 0.5f) + 0.5f) * tDelta;

	// Increment for hitcount heatmap
	int itCount = ceilf(dot(vec3f{ self.sourceRegions[primId].gridSourceBox.size() }, vec3f{ 1.0f, 1.0f, 1.0f }));
	float inc = 1.0f / itCount;
	
	float maxVal = 0;
	float currVal = 0;
	auto bufferOffset = self.sourceRegions[primId].bufferOffset;

	// The incremental phase ("branchless" optiized) from: https://www.shadertoy.com/view/4dX3zl
	vec3i mask;
	float alpha = 0;

	for (int i = 0; i < itCount; i++)
	{
		if (oaabb.contains(localGridPosI))
		{
			
			uint64_t idx = bufferOffset + localGridPosI.x + localGridPosI.y * oaabb.upper.x + localGridPosI.z * oaabb.upper.x * oaabb.upper.y;
			currVal = self.gridData[idx].x;


			float val = tex1D<float4>(optixLaunchParams.transferTexture1D, currVal + optixLaunchParams.transferOffset.x).x;
			alpha += val * optixLaunchParams.inverseIntegralValue.x;

			maxVal = max(currVal, maxVal);
			
			prd.hitCount += inc;
		}
		else
		{
			break;
		}
		mask = ltei(tMax, min(tMax.yzx(), vec3f{ tMax.z, tMax.x, tMax.y }));
		tMax += vec3f(mask) * tDelta;
		localGridPosI += mask * step;
	}
	prd.maxVal = max(prd.maxVal, maxVal);

	vec3f bgColor = { .8f, .8f, .8f };


	vec3f mainColor = spectral_jet(prd.maxVal / self.minmax.y);

	vec3f contentColor = mix(bgColor, mainColor, prd.maxVal / self.minmax.y);
	prd.color = vec4f(contentColor, alpha);
	optixIgnoreIntersection();
}
