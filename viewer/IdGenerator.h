#pragma once
#include <random>

struct IdGenerator
{
	[[nodiscard]] static auto next() noexcept -> unsigned int
	{
		static std::mt19937 gen;
		return gen();
	}
};
