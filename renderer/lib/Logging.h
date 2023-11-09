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

	void log(const std::string& message, LogLevel level = LogLevel::info);
}
