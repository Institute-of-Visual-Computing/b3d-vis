#pragma once

#include <vector>

struct Param
{};

class Application
{
public:
	auto run(const std::vector<Param>& params) -> void;
};
