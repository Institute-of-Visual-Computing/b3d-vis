#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>

#include <string_view>
#include <vector>

#include "ServerClient.h"

class ApplicationSettings
{
public:
	auto save(const std::filesystem::path& filePath) const -> void;
	auto save() const -> void
	{
		save(settingsStorageLocationPath_);
	}

	auto load(const std::filesystem::path& filePath) -> void;
	auto load() -> void;

	static auto restoreDefaultLayoutSettings() -> void;

	std::vector<b3d::tools::project::ServerConnectionDescription> configuredServerSettings_;
	std::filesystem::path settingsStorageLocationPath_ = std::filesystem::current_path() / "app_setting.json";

private:
	NLOHMANN_DEFINE_TYPE_INTRUSIVE(ApplicationSettings, configuredServerSettings_, settingsStorageLocationPath_);
};
