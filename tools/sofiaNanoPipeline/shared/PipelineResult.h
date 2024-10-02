#pragma once

#ifdef B3D_USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

#include "NanoResult.h"
#include "SofiaResult.h"

namespace b3d::tools::sofia_nano_pipeline
{
	struct PipelineResult : common::pipeline::BaseResult
	{
		sofia::SofiaResult sofiaResult;
		nano::NanoResult nanoResult;
	};

	#ifdef B3D_USE_NLOHMANN_JSON
		NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PipelineResult, returnCode, message, finished, sofiaResult, nanoResult);
	#endif
} // namespace b3d::tools::sofia_nano_pipeline
