#pragma once

#ifdef B3D_USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

#include "Result.h"

#include "Vec.h"

namespace b3d::tools::nano
{
	struct NanoResult : common::pipeline::BaseFileResult
	{
		// Size and position of the vdb
		// Due to not existing values for empty space the vdb can be cropped to the bounding box of the data.
		// The bounding box of the original data and the vdb can be different.
		b3d::common::Vec3I voxelSize{};

		// Offset of the vdb in the world space with respect to the original data.
		b3d::common::Vec3I voxelOffset{};

	};

	#ifdef B3D_USE_NLOHMANN_JSON
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NanoResult, returnCode, message, finished, resultFile, fileAvailable, voxelSize,
									   voxelOffset, finishedAt);
	#endif

} // namespace b3d::tools::nano
