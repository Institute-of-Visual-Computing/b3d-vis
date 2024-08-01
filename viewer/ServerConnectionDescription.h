#pragma once

#include <string>
#include <nlohmann/json.hpp>


struct ServerConnectionDescription
{
	std::string port{ "6789" };
	std::string ip{ "localhost" };
	std::string name{"localhost"};
	NLOHMANN_DEFINE_TYPE_INTRUSIVE(ServerConnectionDescription, port, ip, name);
};
