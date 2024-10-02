#pragma once

#include <array>
#include <owl/common.h>

#include "cuda_runtime.h"

#include <optix_types.h>

namespace b3d::renderer
{
	enum class RuntimeVolumeState;

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

	struct CudaSurfaceResource
	{
		cudaArray_t buffer;
		cudaSurfaceObject_t surface;
		size_t width{};
		size_t height{};
	};

	struct CudaStereoRenderTarget
	{
		std::array<CudaSurfaceResource, 2> surfaces;
		Extent extent;
	};

	struct ExternalRenderTarget
	{
		cudaGraphicsResource_t target;
		Extent extent;
		void* nativeHandle{ nullptr };
	};

	// ExternalTexture is read only
	using ExternalTexture = ExternalRenderTarget;

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
		owl::box3f volumeVoxelBox{};
		owl::AffineSpace3f renormalizedScale = owl::AffineSpace3f::scale(owl::vec3f{ 1, 1, 1 });
	};

	struct RendererInitializationInfo
	{
		cudaUUID_t deviceUuid;
		int deviceIndex;
	};

	enum class ColoringMode
	{
		single = 0,
		colormap = 1
	};

	struct ColorRGB
	{
		float r, g, b;
	};

	struct ColorRGBA
	{
		float r, g, b, a;
	};

	struct ColoringInfo
	{
		ColoringMode coloringMode;
		owl::vec4f singleColor;
		float selectedColorMap;
		std::array<ColorRGBA, 2> backgroundColors;
	};

	struct ColorMapInfos
	{
		std::vector<std::string>* colorMapNames{};
		int colorMapCount;
		float firstColorMapYTextureCoordinate;
		float colorMapHeightNormalized;
	};

	struct FoveatedRenderingControl
	{
		owl::vec2f leftEyeGazeScreenSpace{ 0.0f, 0.0f };
		owl::vec2f rightEyeGazeScreenSpace{ 0.0f, 0.0f };
		bool isEnabled{ false };
		float temporalBufferResolutionRelativeScale{ 1.0f };
		float kernelParameter{ 1.0f };
	};
	
	struct NanoVdbVolume
	{
		owl::box3f indexBox{};
		owl::box3f worldAabb{};
		owl::AffineSpace3f transform{};
		CUdeviceptr grid = 0;
	};

	struct FitsNanoVdbVolume
	{
		CUdeviceptr grid = 0;
	};

	enum class RuntimeVolumeState
	{
		loadingRequested,
		ready,
		unloadedRequested,
		unloaded
	};

	struct FitsNanoRuntimeVolume
	{
		FitsNanoVdbVolume volume{};
		RuntimeVolumeState state{};
		owl::AffineSpace3f renormalizeScale{};
		// Stores the AABB of the volume in world space
		owl::box3f fitsIndexBox{};
		// Stores the AABB of the potentially shrinked nvdb-grid volume in index space. Must be smaller than the
		// fitsIndexBox.
		owl::box3f indexBox{};
	};

	struct RuntimeVolume
	{
		NanoVdbVolume volume{};
		RuntimeVolumeState state{};
		owl::AffineSpace3f renormalizeScale{};
		std::string uuid{};
	};

	struct RuntimeVolumeData
	{
		bool newVolumeAvailable{ false };
		RuntimeVolume volume{};
		bool newProjectAvailable{ false };
		owl::box3f originalIndexBox{};
	};

}
