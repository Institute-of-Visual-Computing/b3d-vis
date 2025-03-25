#pragma once
#include <owl/common.h>

#include "Common.h"

#include "SharedRenderingStructs.h"

namespace b3d
{
	namespace renderer
	{
		namespace fitsNvdb
		{

			struct FitsNvdbGeometry
			{
				// Excluding max.
				owl::box3f fitsBox{};
				// Excluding max.
				owl::box3f nvdbBox{};
			};

			struct RayGenerationData
			{
				owl::vec2i frameBufferSize;
				OptixTraversableHandle world;
			};

			struct RayCameraData
			{
				// Origin of the camera in world space
				owl::vec3f pos;
				//
				owl::vec3f dir_00;
				//
				owl::vec3f dir_du;
				//
				owl::vec3f dir_dv;
			};

			enum class SampleIntegrationMethod
			{
				transferIntegration,
				maximumIntensityProjection,
				averageIntensityProjection
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

				tools::renderer::nvdb::FitsNanoVdbVolume volume;
			};
		} // namespace fitsNvdb
	} // namespace renderer
} // namespace b3d
