#pragma once

#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include <optix_types.h>
#include <surface_types.h>


namespace b3d
{
	namespace renderer
	{
		namespace nano
		{
			struct LaunchParams
			{
				uint32_t outputSurfaceIndex;
			};

			struct NanoVdbVolume
			{
				owl::box3f indexBox;
				owl::box3f worldAabb;
				owl::AffineSpace3f transform;
				CUdeviceptr grid = 0;
			};

			struct Camera
			{
				owl::vec3f position;
				owl::vec3f dir00;
				owl::vec3f dirDu;
				owl::vec3f dirDv;
			};

			struct Volume
			{
				void* grid;
			};

			struct GeometryData
			{
				NanoVdbVolume volume;
			};

			struct RayGenerationData
			{
				uint32_t* frameBufferPtr;
				owl::vec2i frameBufferSize;
				OptixTraversableHandle world;
				Camera camera;
				cudaSurfaceObject_t* outputSurfaceArray;
			};

			struct MissProgramData
			{
				owl::vec3f color0;
				owl::vec3f color1;
			}; // namespace nano
		} // namespace nano
	} // namespace renderer
} // namespace b3d
