#pragma once

#include <chrono>

namespace b3d::common::helper
{
	inline auto getSecondsSinceEpochUtc() -> long long
	{
		auto now = std::chrono::utc_clock::now();
		return std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch().count();
	}

	inline auto getTimePointFromUtcSeconds(long long utcSecondsSinceEpoch)
		-> std::chrono::time_point<std::chrono::utc_clock>
	{
		std::chrono::seconds dur(utcSecondsSinceEpoch);
		return std::chrono::time_point<std::chrono::utc_clock>(dur);
	}

} // namespace b3d::common::helper
