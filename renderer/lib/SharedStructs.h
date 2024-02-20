#pragma once

#include <array>
#include <owl/common.h>

#include "cuda_runtime.h"

namespace b3d::renderer
{
	struct Camera
	{
		owl::vec3f origin;
		owl::vec3f at;
		owl::vec3f up;
		float cosFoV;
		float FoV; // in radians
		bool directionsAvailable{ false };
		owl::vec3f dir00;
		owl::vec3f dirDu;
		owl::vec3f dirDv;
	};

	struct Extent
	{
		uint32_t width;
		uint32_t height;
		uint32_t depth;
	};

	struct ExternalRenderTarget
	{
		cudaGraphicsResource_t target;
		Extent extent;
	};

	enum class RenderMode : int
	{
		mono = 0,
		stereo
	};

	struct View
	{
		std::array<Camera, 2> cameras;
		RenderMode mode;
	};

	struct RenderTargets
	{
		ExternalRenderTarget colorRt;
		ExternalRenderTarget minMaxRt;
	};

	struct Synchronization
	{
		cudaExternalSemaphore_t waitSemaphore;
		cudaExternalSemaphore_t signalSemaphore;
		uint64_t fenceValue{ 0 };
	};
	
	struct VolumeTransform
	{
		owl::affine3f worldMatTRS{};
	};

	struct RendererInitializationInfo
	{
		cudaUUID_t deviceUuid;
		int deviceIndex;
	};

	enum ColoringMode
	{
		single = 0,
		colormap = 1
	};

	struct ColoringInfo
	{
		ColoringMode coloringMode;
		owl::vec4f singleColor;
		float selectedColorMap;
	};
}
