#pragma once

#include <string>

namespace b3d
{
	enum class LogLevel
	{
		info,
		warning,
		error
	};

	auto log(const std::string& message, LogLevel level = LogLevel::info) -> void;
} // namespace b3d
