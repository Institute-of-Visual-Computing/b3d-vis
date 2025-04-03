#pragma once

#include <chrono>
#include <format>

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

	// https://stackoverflow.com/a/68754043
	// https://creativecommons.org/licenses/by-sa/4.0/
	inline auto getNowAsFormattedDateTimeString(const std::string formatString) -> std::string
	{
		auto const time = std::chrono::current_zone()->to_local(std::chrono::system_clock::now());
		return std::format("{:%d.%m.%Y %H\:%M}", time);
	}

} // namespace b3d::common::helper
