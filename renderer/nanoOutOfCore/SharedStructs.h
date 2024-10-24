#pragma once

#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include <optix_types.h>
#include <surface_types.h>
#include "SharedRenderingStructs.h"

namespace b3d
{
	namespace renderer
	{
		namespace nano
		{
			struct RayCameraData
			{
				owl::vec3f pos;
				owl::vec3f dir_00;
				owl::vec3f dir_du;
				owl::vec3f dir_dv;
			};

			struct LaunchParams
			{
				RayCameraData cameraData;
				cudaSurfaceObject_t surfacePointer;
				struct BG
				{
					owl::vec3f color0;
					owl::vec3f color1;
					bool fillBox;
					owl::vec3f fillColor;
				} bg;
				owl::vec3f color;
			};

			struct Volume
			{
				void* grid;
			};

			struct GeometryData
			{
				tools::renderer::nvdb::NanoVdbVolume volume;
			};

			struct RayGenerationData
			{
				owl::vec2i frameBufferSize;
				OptixTraversableHandle world;
			};

			struct MissProgramData
			{
				owl::vec3f color0;
				owl::vec3f color1;
			}; // namespace nano
		} // namespace nano
	} // namespace renderer
} // namespace b3d
