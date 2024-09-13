#pragma once

#include <filesystem>
#include <string>

#ifdef B3D_USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

#include "Box.h"
#include "PipelineResult.h"
#include "SofiaParams.h"

namespace b3d::tools::project
{
	struct Request {

		std::string uuid;

		// Region the request is defined on. Inside original fits volume.
		// lower inclusive, upper exclusive -> size = upper - lower
		common::Box3I subRegion;

		// SofiaParams
		sofia::SofiaParams sofiaParameters;
		b3d::tools::sofia_nano_pipeline::PipelineResult result;

		#ifdef B3D_USE_NLOHMANN_JSON
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(Request, uuid, sofiaParameters, result, subRegion);
		#endif

		auto createUUID() const -> std::string;

	};
}
