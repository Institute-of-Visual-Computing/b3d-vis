#include "Logging.h"
#include <iostream>

auto b3d::log(const std::string& message, const LogLevel level) -> void
{
	switch (level)
	{
	case b3d::LogLevel::info:
		std::cout << "[INFO]: ";
		break;
	case b3d::LogLevel::warning:
		std::cout << "[WARNING]: ";
		break;
	case b3d::LogLevel::error:
		std::cout << "[ERROR]: ";
		break;
	}

	std::cout << message << std::endl;
}
