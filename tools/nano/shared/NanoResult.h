#pragma once

#ifdef B3D_USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

#include "Result.h"

namespace b3d::tools::nano
{
	struct NanoResult : common::pipeline::BaseFileResult
	{
	};

	#ifdef B3D_USE_NLOHMANN_JSON
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NanoResult, returnCode, message, finished, resultFile, fileAvailable);
	#endif

} // namespace b3d::tools::nano
