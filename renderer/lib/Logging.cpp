#include "Logging.h"
#include <iostream>

void b3d::log(const std::string& message, LogLevel level)
{
	switch(level)
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
