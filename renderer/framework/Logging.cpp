#include "Logging.h"
#include <iostream>

using namespace b3d::renderer;

auto b3d::renderer::log(const std::string& message, const LogLevel level) -> void
{
	switch (level)
	{
	case LogLevel::info:
		std::cout << "[INFO]: ";
		break;
	case LogLevel::warning:
		std::cout << "[WARNING]: ";
		break;
	case LogLevel::error:
		std::cout << "[ERROR]: ";
		break;
	}

	std::cout << message << std::endl;
}
