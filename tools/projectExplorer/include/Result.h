#pragma once

#include "nlohmann/json.hpp"

namespace b3d::tools::projectexplorer
{
	struct SingleResult
	{
		int returnCode{ -1 };
		std::string message{ "" };
		bool finished{ false };
		std::string fileResultGUID{ "" };

		NLOHMANN_DEFINE_TYPE_INTRUSIVE(SingleResult, returnCode, message, fileResultGUID, finished)

		auto wasSuccess() const -> bool
		{
			return finished && returnCode == 0;
		}
	};

	struct RequestResult
	{
		int returnCode{ -1 };
		std::string message{""};
		bool finished{ false };
		SingleResult sofia;
		SingleResult nvdb;

		NLOHMANN_DEFINE_TYPE_INTRUSIVE(RequestResult, returnCode, message, sofia, nvdb, finished)

		auto wasSuccess() const -> bool
		{
			return finished && returnCode == 0;
		}
	};
}


