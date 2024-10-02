#pragma once
#include <string>

#ifdef B3D_USE_NLOHMANN_JSON
	#include <nlohmann/json.hpp>
#endif

namespace b3d::common::pipeline
{
	struct BaseResult
	{
		int returnCode{ -1 };
		std::string message{ "" };
		bool finished{ false };

		// Timestamp of finish. Since Epoch in seconds.
		long long finishedAt;

		auto wasSuccess() const -> bool
		{
			return finished && returnCode == 0;
		}
	};

	struct BaseFileResult : BaseResult
	{
		// either a UUID or a path
		std::string resultFile{ "" };
		bool fileAvailable{ true };
	};

	#ifdef B3D_USE_NLOHMANN_JSON
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BaseResult, returnCode, message, finished, finishedAt);
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BaseFileResult, returnCode, message, finished, resultFile, fileAvailable,
									   finishedAt);
	#endif
} // namespace b3d::tools::common::pipeline
