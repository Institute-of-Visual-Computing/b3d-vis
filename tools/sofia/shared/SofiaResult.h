#pragma once

#include <Result.h>

#ifdef B3D_USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace b3d::tools::sofia
{
	inline constexpr const char* sofia_return_code_messages[9] = {
		"The pipeline successfully completed without any error.",
		"An unclassified failure occurred.",
		"A NULL pointer was encountered.",
		"A memory allocation error occurred. This could indicate that the data cube is too large for the amount of "
		"memory available on the machine.",
		"An array index was found to be out of range.",
		"An error occurred while trying to read or write a file or check if a directory or file is accessible.",
		"The overflow of an integer value occurred.",
		"The pipeline had to be aborted due to invalid user input. This could, e.g., be due to an invalid parameter "
		"setting or the wrong input file being provided.",
		"No specific error occurred, but sources were not detected either."
	};

	struct SofiaResult : common::pipeline::BaseFileResult
	{
		auto getSofiaResultMessage() const -> std::string
		{
			if (0 > returnCode || returnCode >= 9)
			{
				return sofia_return_code_messages[1];
			}
			return sofia_return_code_messages[returnCode];
		}
	};

#ifdef B3D_USE_NLOHMANN_JSON
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SofiaResult, returnCode, message, finished, resultFile, fileAvailable,
									   finishedAt);
#endif

} // namespace b3d::tools::sofia
