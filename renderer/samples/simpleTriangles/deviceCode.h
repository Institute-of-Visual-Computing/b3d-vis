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

#pragma once

#include "owl/Object.h"
#include <owl/common/math/vec.h>
#include <owl/owl.h>

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
};

/* variables for the triangle mesh geometry */
struct TrianglesGeomData
{
	/*! base color we use for the entire mesh */
	vec4f color;
	/*! array/buffer of vertex indices */
	vec3i* index;
	/*! array/buffer of vertex positions */
	vec3f* vertex;

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
