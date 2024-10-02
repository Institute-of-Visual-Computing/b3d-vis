// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>

#include "Common.h"
#include "deviceCode.h"

#include "owl/owl_device.h"
#include "../../nano/OptixHelper.cuh"

namespace
{
	#define PRINT_VEC3F(name, vec)                                                                                         \
	{                                                                                                                  \
		printf("%s: x: %.5f, y: %.5f, z: %.5f\n", name, (float)(vec).x, (float)(vec).y, (float)(vec).z);               \
	}

	#define PRINT_VEC4F(name, vec)                                                                                         \
	{                                                                                                                  \
		printf("%s: x: %.5f, y: %.5f, z: %.5f, w: %.5f\n", name, (float)(vec).x, (float)(vec).y, (float)(vec).z, (float)(vec).w);               \
	}

#define PRINT_VEC3I(name, vec)                                                                                         \
	{                                                                                                                  \
		printf("%s: x: %d, y: %d, z: %d\n", name, (int)(vec).x, (int)(vec).y, (int)(vec).z);                           \
	}

#define PRINT_NEWLINE() printf("\n");
#define PRINT_HORIZ_LINE() printf("------------\n");
}

__constant__ MyLaunchParams optixLaunchParams;

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
	const RayGenData& self = owl::getProgramData<RayGenData>();
	const vec2i pixelID = owl::getLaunchIndex();

	const vec2f screen = (vec2f(pixelID) + vec2f(.5f)) / vec2f(self.fbSize);
	owl::Ray ray;
	ray.origin = optixLaunchParams.cameraData.pos;
	ray.direction = normalize(optixLaunchParams.cameraData.dir_00 + screen.u * optixLaunchParams.cameraData.dir_du +
							  screen.v * optixLaunchParams.cameraData.dir_dv);
	PerRayData prd;
	prd.color = 0;
	prd.frameBufferSize = self.fbSize;

	
	owl::traceRay(/*accel to trace against*/ self.world,
				  /*the ray to trace*/ ray,
				  /*prd*/ prd);

	//const int fbOfs = pixelID.x + self.fbSize.x * pixelID.y;
	surf2Dwrite(owl::make_rgba(prd.color), optixLaunchParams.surfacePointer, sizeof(uint32_t) * pixelID.x, pixelID.y);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
	PerRayData& prd = owl::getPRD<PerRayData>();

	const TrianglesGeomData& self = owl::getProgramData<TrianglesGeomData>();

	// compute normal:
	const int primID = optixGetPrimitiveIndex();
	const vec3i index = self.index[primID];
	const vec3f& A = self.vertex[index.x];
	const vec3f& B = self.vertex[index.y];
	const vec3f& C = self.vertex[index.z];
	const vec3f Ng = normalize(cross(B - A, C - A));

	const vec3f rayDir = optixGetWorldRayDirection();

	const vec2f uv = optixGetTriangleBarycentrics();
	const vec2f tc =
		(1.f - uv.x - uv.y) * self.texCoord[index.x] + uv.x * self.texCoord[index.y] + uv.y * self.texCoord[index.z];


	prd.color = (.2f + .8f * fabs(dot(rayDir, Ng)));
	if (optixLaunchParams.coloringInfo.coloringMode == b3d::renderer::ColoringMode::single)
	{
		prd.color *= optixLaunchParams.coloringInfo.singleColor;
	}
	else
	{
		vec4f bla = tex2D<float4>(optixLaunchParams.colorMaps, tc.y, optixLaunchParams.coloringInfo.selectedColorMap);
		prd.color *= bla;
	}
}

OPTIX_MISS_PROGRAM(miss)()
{
	const vec2i pixelID = owl::getLaunchIndex();

	const MissProgData& self = owl::getProgramData<MissProgData>();

	PerRayData& prd = owl::getPRD<PerRayData>();
	int pattern = (pixelID.x / 8) ^ (pixelID.y / 8);
	prd.color = vec4f{ (pattern & 1) ? optixLaunchParams.backgroundColor0 : optixLaunchParams.backgroundColor1 };
}
