#pragma once

#include "nlohmann/json.hpp"

#include "Request.h"

namespace b3d::tools::projectexplorer
{
	struct Project
	{
		std::string b3dViewerProjectVersion;
		std::string projectName;
		std::string projectUUID;
		std::string fitsOriginGUID;
		std::filesystem::path projectPathAbsolute;
		std::vector<Request> requests;

		NLOHMANN_DEFINE_TYPE_INTRUSIVE(Project, b3dViewerProjectVersion, projectName, projectUUID, fitsOriginGUID,
									   requests);
	};

	inline auto operator==(const b3d::tools::projectexplorer::Project& x,
	                       const b3d::tools::projectexplorer::Project& y) -> bool
	{
		return x.b3dViewerProjectVersion == y.b3dViewerProjectVersion && x.fitsOriginGUID == y.fitsOriginGUID &&
			x.projectUUID == y.projectUUID;
	}
}

namespace std
{
	template <>
	struct hash<b3d::tools::projectexplorer::Project>
	{
		typedef b3d::tools::projectexplorer::Project argument_type;
		typedef size_t result_type;

		auto operator()(const b3d::tools::projectexplorer::Project& x) const noexcept -> size_t
		{
			return filesystem::hash_value(x.fitsOriginGUID);
		}
	};
} // namespace std
