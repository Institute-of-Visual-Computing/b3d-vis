#pragma once

#include <owl/common/math/vec.h>
#include <owl/owl.h>
#include "owl/Object.h"


using namespace owl;

struct RayCameraData
{
	vec3f pos;
	vec3f dir_00;
	vec3f dir_du;
	vec3f dir_dv;
};

struct MyLaunchParams
{
	RayCameraData cameraData;
	cudaSurfaceObject_t surfacePointer;
	cudaTextureObject_t transferTexture1D;
	float1 transferOffset{ 0 };
	float1 integralValue{ 0 };
	float1 inverseIntegralValue{ 0 };

};

struct SourceRegion
{
	owl::box3i gridSourceBox{ { 0 }, { 1 } };
	owl::box3f sourceBoxNormalized{ { 0 }, { 1 } };
	uint64_t bufferOffset{ 0 };
};

struct DatacubeSources
{
	SourceRegion* sourceRegions;
	float1* gridData;
	vec3i gridDims;
	vec2f minmax;
};

struct VolumeGridData
{
	float1* gridData;
	vec3i gridDims;
};

/* variables for the ray generation program */
struct RayGenData
{
	vec2i fbSize;
	OptixTraversableHandle world;
};

/* variables for the miss program */
struct MissProgData
{
	vec4f color0;
	vec4f color1;
};

