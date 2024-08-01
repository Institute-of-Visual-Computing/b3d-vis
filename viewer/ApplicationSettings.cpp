#include "ApplicationSettings.h"
#include <fstream>
#include <print>

auto ApplicationSettings::save(const std::filesystem::path& filePath) const -> void
{
	const auto j = nlohmann::json(*this);
	std::ofstream o(filePath);
	o << std::setw(4) << j << std::endl;
}

auto ApplicationSettings::load(const std::filesystem::path& filePath) -> void
{
	assert(std::filesystem::exists(filePath));
	std::ifstream f(filePath);

	try
	{
		const auto data = nlohmann::json::parse(f);
		*this = data.get<ApplicationSettings>();
	}
	catch (nlohmann::json::type_error& e)
	{
		std::println("[settings]: {}", e.what());
	}
}
