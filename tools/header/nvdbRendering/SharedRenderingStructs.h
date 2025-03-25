#pragma once

#include "cuda.h"
#include "owl/common/math/affinespace.h"

namespace b3d::tools::renderer::nvdb
 {
	enum class RuntimeVolumeState
	{
		loadingRequested,
		ready,
		unloadedRequested,
		unloaded
	};

	struct NanoVdbVolume
	{
		owl::box3f indexBox{};
		owl::box3f worldAabb{};
		owl::AffineSpace3f transform{};
		CUdeviceptr grid = 0;
	};

	struct RuntimeVolume
	{
		NanoVdbVolume volume{};
		RuntimeVolumeState state{};
		owl::AffineSpace3f renormalizeScale{};
		std::string uuid{};
	};

	struct FitsNanoVdbVolume
	{
		CUdeviceptr grid = 0;
	};

	struct FitsNanoRuntimeVolume
	{
		FitsNanoVdbVolume volume{};
		tools::renderer::nvdb::RuntimeVolumeState state{};
		owl::AffineSpace3f renormalizeScale{};
		// Stores the AABB of the volume in world space
		owl::box3f fitsIndexBox{};
		// Stores the AABB of the potentially shrinked nvdb-grid volume in index space. Must be smaller than the
		// fitsIndexBox.
		owl::box3f indexBox{};
	};

	struct RuntimeVolumeData
	{
		bool newVolumeAvailable{ false };
		b3d::tools::renderer::nvdb::RuntimeVolume volume{};
		bool newProjectAvailable{ false };
		owl::box3f originalIndexBox{};
	};
 }
