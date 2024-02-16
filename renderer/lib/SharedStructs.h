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
		owl::vec3f position{ 0, 0, 0 };
		owl::vec3f scale{ 1, 1, 1 };
		owl::Quaternion3f rotation{ 1 };
		owl::affine3f worldMatTRS{};
	};

	struct RendererState
	{
		owl::affine3f worldMatTRS{};
	};

	struct RendererInitializationInfo
	{
		cudaUUID_t deviceUuid;
		int deviceIndex;
	};


}
