#pragma once

#include <string_view>

namespace b3d::profiler
{
	struct ProfilerResult
	{
		std::string_view name;
		float start;
		float stop;
	};
} // namespace b3d::profiler
