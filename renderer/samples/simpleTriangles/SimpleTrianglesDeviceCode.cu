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
#include "deviceCode.h"
#include "owl/owl_device.h"


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

	vec4f color;
	owl::traceRay(/*accel to trace against*/ self.world,
				  /*the ray to trace*/ ray,
				  /*prd*/ color);

	const int fbOfs = pixelID.x + self.fbSize.x * pixelID.y;
	surf2Dwrite(owl::make_rgba(color), optixLaunchParams.surfacePointer, sizeof(uint32_t) * pixelID.x, pixelID.y);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
	vec4f& prd = owl::getPRD<vec4f>();

	const TrianglesGeomData& self = owl::getProgramData<TrianglesGeomData>();

	// compute normal:
	const int primID = optixGetPrimitiveIndex();
	const vec3i index = self.index[primID];
	const vec3f& A = self.vertex[index.x];
	const vec3f& B = self.vertex[index.y];
	const vec3f& C = self.vertex[index.z];
	const vec3f Ng = normalize(cross(B - A, C - A));

	const vec3f rayDir = optixGetWorldRayDirection();
	prd = (.2f + .8f * fabs(dot(rayDir, Ng))) * self.color;
	prd.w = 1.0f;
}

OPTIX_MISS_PROGRAM(miss)()
{
	const vec2i pixelID = owl::getLaunchIndex();

	const MissProgData& self = owl::getProgramData<MissProgData>();

	vec4f& prd = owl::getPRD<vec4f>();
	int pattern = (pixelID.x / 8) ^ (pixelID.y / 8);
	prd = 0;
	
}
