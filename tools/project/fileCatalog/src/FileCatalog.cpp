#include <fstream>
#include <iostream>
#include <ranges>

#include <uuid.h>
#include <nlohmann/json.hpp>

#include "FileCatalog.h"



static uuids::uuid_name_generator gNameGenerator{ uuids::uuid::from_string("123456789-abcdef-123456789-abcdef-12").value() };

b3d::tools::project::catalog::FileCatalog::FileCatalog(const std::filesystem::path& rootPath) : rootPath_(rootPath)
{
}

b3d::tools::project::catalog::FileCatalog::FileCatalog(const std::filesystem::path& rootPath,
	const std::filesystem::path& dataPath) : rootPath_(rootPath), dataPath_(dataPath)
{
}

// Absolute Path!
auto b3d::tools::project::catalog::FileCatalog::addFilePathAbsolute(
	const std::filesystem::path& absoluteFilePath)
	->const std::string
{
	const auto relativePath = absoluteFilePath.lexically_relative(rootPath_);
	return addFilePathRelativeToRoot(relativePath);
}

auto b3d::tools::project::catalog::FileCatalog::addFilePathRelativeToRoot( 
	const std::filesystem::path& relativeFilePath)
	-> const std::string
{
	auto fileUUID = uuids::to_string(gNameGenerator(relativeFilePath.generic_string()));
	FileCatalogEntry ce{ relativeFilePath, relativeFilePath.filename() };
	
	mappings_.insert(std::make_pair(fileUUID, ce));
	return fileUUID;
}

auto b3d::tools::project::catalog::FileCatalog::addFilePathAbsoluteWithUUID(const std::filesystem::path& filePath,
	const std::string& fileUUID) -> void
{
	const auto relativeFilePath = filePath.lexically_relative(rootPath_);
	FileCatalogEntry ce{ relativeFilePath, relativeFilePath.filename() };
	mappings_.insert(std::make_pair(fileUUID, ce));
}

auto b3d::tools::project::catalog::FileCatalog::contains(const std::string& fileUUID) const -> bool
{
	return mappings_.contains(fileUUID);
}

auto b3d::tools::project::catalog::FileCatalog::getFilePathRelativeToRoot(
	const std::string& fileUUID) const -> const std::filesystem::path
{
	if (mappings_.contains(fileUUID))
	{
		return mappings_.at(fileUUID).filePath;
	}
	return {};
}


auto b3d::tools::project::catalog::FileCatalog::getFilePathAbsolute(
	const std::string& fileUUID) const -> const std::filesystem::path
{
	if (mappings_.contains(fileUUID))
	{
		return rootPath_ / mappings_.at(fileUUID).filePath;
	}
	return {};
}

auto b3d::tools::project::catalog::FileCatalog::getMappings() const -> const std::map<std::string, FileCatalogEntry>&
{
	return mappings_;
}

auto b3d::tools::project::catalog::FileCatalog::removeMapping(const std::string& fileUUID) -> void
{
	mappings_.erase(fileUUID);
}

auto b3d::tools::project::catalog::FileCatalog::relativeMappings() -> void
{
	for (auto& entry : mappings_ | std::views::values | std::views::all)
	{
		if (entry.filePath.is_absolute())
		{
			entry.filePath = entry.filePath.lexically_relative(rootPath_);
		}
	}
}

auto b3d::tools::project::catalog::FileCatalog::removeInvalidMappings() -> void
{
	std::vector<std::string> identifiersToRemove{};

	for (const auto& kv : mappings_)
	{
		if (!exists(rootPath_ / kv.second.filePath))
		{
			identifiersToRemove.push_back(kv.first);
		}
	}

	for (const auto& uuid : identifiersToRemove)
	{
		removeMapping(uuid);
	}
}

auto b3d::tools::project::catalog::FileCatalog::setRootPath(const std::filesystem::path& rootPath) -> void
{
	rootPath_ = rootPath;
}

auto b3d::tools::project::catalog::FileCatalog::getRootPath() const -> const std::filesystem::path
{
	return rootPath_;
}

auto b3d::tools::project::catalog::FileCatalog::getDataPathAbsolute() const -> const std::filesystem::path
{
	return rootPath_ / dataPath_;
}

b3d::tools::project::catalog::FileCatalog b3d::tools::project::catalog::FileCatalog::createOrLoadCatalogFromPathes(
	const std::filesystem::path& absoluteRootPath, const std::filesystem::path& relativeDataPath)
{
	if (!std::filesystem::exists(absoluteRootPath))
	{
		std::filesystem::create_directories(absoluteRootPath);
	}
	if (!std::filesystem::exists(absoluteRootPath / relativeDataPath))
	{
		std::filesystem::create_directories(absoluteRootPath / relativeDataPath);
	}

	// Check if catalog already exists in dataRootPath. Parse catalog if it exists otherwise create a new one.
	const auto catalogFilePath = absoluteRootPath / "fileCatalog.json";
	FileCatalog catalog{ absoluteRootPath, relativeDataPath };
	if (std::filesystem::exists(catalogFilePath) && !std::filesystem::is_directory(catalogFilePath))
	{
		try
		{
			std::ifstream f(catalogFilePath);

			const auto data = nlohmann::json::parse(f);
			auto c = data.get<project::catalog::FileCatalog>();
			c.removeInvalidMappings();
			catalog = c;
		}
		catch (nlohmann::json::type_error& e)
		{
			// std::cout << e.what();
			// [json.exception.type_error.304] cannot use at() with object
		}
		catch (nlohmann::json::parse_error& e)
		{
			// std::cout << e.what();
		}
		catch (nlohmann::json::exception& e)
		{
			// std::cout << e.what();
		}

		std::ifstream fileStream{ catalogFilePath };
		nlohmann::json j;
		fileStream >> j;
		catalog = j.get<FileCatalog>();
	}
	else
	{
		catalog = FileCatalog{ absoluteRootPath, relativeDataPath };
	}
	return catalog;
}
