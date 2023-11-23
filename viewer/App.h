#pragma once

#include <string>
#include <vector>

struct Param
{
	std::string value;
};

class Application
{
public:
	auto run() -> void;
	auto initialization(const std::vector<Param>& vector) -> void;
};
