#pragma once

#include <string>
#include <vector>

namespace b3d::tools::common::fits
{
	struct FitsProperties
	{
		int axisCount;
		int imgType;
		std::vector<long> axisDimensions;
		std::vector<std::string> axisTypes;
		std::string unit;
	};

} // namespace b3d::tools::common::fits
