#pragma once

#include <string>

#include <filesystem>

#include "Result.h"
#include "nlohmann/json.hpp"

namespace b3d::tools::projectexplorer
{
	struct Request {

		std::string guid;
		std::vector<std::string> preSofiaSearchParameters;
		std::filesystem::path sofiaExecutablePath;
		std::filesystem::path workingDirectory;

		std::vector<std::string> sofiaSearchParameters;
		
		RequestResult result;

		NLOHMANN_DEFINE_TYPE_INTRUSIVE(Request, guid, sofiaSearchParameters, result)

		auto createUUID() const -> std::string;

	};
}
