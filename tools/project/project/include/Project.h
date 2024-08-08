#pragma once

#include "nlohmann/json.hpp"

#include "Request.h"

#include "FitsCommon.h"

#ifdef B3D_USE_NLOHMANN_JSON
	#include "nlohmann/json.hpp"
#endif

namespace b3d::tools::project
{
		struct Project
		{
			std::string b3dViewerProjectVersion;
			std::string projectName;
			std::string projectUUID;
			std::string fitsOriginUUID;
			std::string fitsOriginFileName;
			b3d::tools::fits::FitsProperties fitsOriginProperties;
			std::vector<b3d::tools::project::Request> requests;
			std::filesystem::path projectPathAbsolute;

			#ifdef B3D_USE_NLOHMANN_JSON
				NLOHMANN_DEFINE_TYPE_INTRUSIVE(Project, b3dViewerProjectVersion, projectName, projectUUID, fitsOriginUUID, fitsOriginFileName, fitsOriginProperties, requests);
			#endif
		};

		inline auto operator==(const b3d::tools::project::Project& x, const b3d::tools::project::Project& y) -> bool
		{
			return x.projectUUID == y.projectUUID;
		}
} // namespace b3d::tools::project

