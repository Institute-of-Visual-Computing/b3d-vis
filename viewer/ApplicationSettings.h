#pragma once

#include "ServerConnectionDescription.h"

#include <filesystem>
#include <nlohmann/json.hpp>


#include <vector>

class ApplicationSettings
{

public:
	auto save(const std::filesystem::path& filePath) const -> void;
	auto save() const -> void
	{
		save(settingsStorageLocationPath_);
	}

	auto load(const std::filesystem::path& filePath) -> void;
	auto load() -> void
	{
		load(settingsStorageLocationPath_);
	}


	std::vector<ServerConnectionDescription> configuredServerSettings_;
	std::filesystem::path settingsStorageLocationPath_ = std::filesystem::current_path() / "app_setting.json";
private:
	NLOHMANN_DEFINE_TYPE_INTRUSIVE(ApplicationSettings, configuredServerSettings_, settingsStorageLocationPath_);
};
