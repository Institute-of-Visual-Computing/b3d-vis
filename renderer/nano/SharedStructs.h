#pragma once

#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include <optix_types.h>
#include <surface_types.h>
#include <Common.h>


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

			enum class SampleIntegrationMethod
			{
				transferIntegration,
				maximumIntensityProjection,
				averageIntensityProjection
			};

			struct NanoVdbVolume
			{
				owl::box3f indexBox;
				owl::box3f worldAabb;
				owl::AffineSpace3f transform;
				CUdeviceptr grid = 0;
			};

			struct LaunchParams
			{
				RayCameraData cameraData;
				cudaSurfaceObject_t surfacePointer;
				struct BG
				{
					owl::vec4f color0;
					owl::vec4f color1;
					bool fillBox;
					owl::vec3f fillColor;
				} bg;
				cudaTextureObject_t colorMaps;
				ColoringInfo coloringInfo;
				cudaTextureObject_t transferFunctionTexture;
				owl::vec2f sampleRemapping;
				SampleIntegrationMethod sampleIntegrationMethod;
				NanoVdbVolume volume;
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
