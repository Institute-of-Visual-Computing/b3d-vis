#include "FileCatalog.h"

#include <fstream>
#include <iostream>
#include <ranges>

#include <nlohmann/json.hpp>
#include <uuid.h>

static uuids::uuid_name_generator gNameGenerator{
	uuids::uuid::from_string("123456789-abcdef-123456789-abcdef-12").value()
};

b3d::tools::project::catalog::FileCatalog::FileCatalog(const std::filesystem::path& rootPath,
													   const std::string& filename)
	: rootPath_(rootPath), fileName_(filename)
{
}

// Absolute Path!
auto b3d::tools::project::catalog::FileCatalog::addFilePathAbsolute(const std::filesystem::path& absoluteFilePath,
																	bool relativizePath) -> const std::string
{
	if (relativizePath)
	{
		const auto relativePath = absoluteFilePath.lexically_relative(rootPath_);
		return addFilePathRelativeToRoot(relativePath);
	}
	return addFilePathRelativeToRoot(absoluteFilePath);
}

auto b3d::tools::project::catalog::FileCatalog::addFilePathRelativeToRoot(const std::filesystem::path& relativeFilePath)
	-> const std::string
{
	auto fileUUID = uuids::to_string(gNameGenerator(relativeFilePath.generic_string()));
	if (mappings_.contains(fileUUID))
	{
		return fileUUID;
	}
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

auto b3d::tools::project::catalog::FileCatalog::getFilePathRelativeToRoot(const std::string& fileUUID) const
	-> const std::filesystem::path
{
	if (mappings_.contains(fileUUID))
	{
		return mappings_.at(fileUUID).filePath;
	}
	return {};
}


auto b3d::tools::project::catalog::FileCatalog::getFilePathAbsolute(const std::string& fileUUID) const
	-> const std::filesystem::path
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

auto b3d::tools::project::catalog::FileCatalog::getRootPath() const -> const std::filesystem::path
{
	return rootPath_;
}

auto b3d::tools::project::catalog::FileCatalog::setRootPath(const std::filesystem::path& rootPath) -> void
{
	rootPath_ = rootPath;
}

auto b3d::tools::project::catalog::FileCatalog::setFileName(const std::string& fileName) -> void
{
	fileName_ = fileName;
}

auto b3d::tools::project::catalog::FileCatalog::writeCatalog() const -> void
{
	const auto catalogFilePath = rootPath_ / DEFAULT_CATALOG_FILE_NAME;
	std::ofstream fileStream{ catalogFilePath };
	nlohmann::json j = *this;
	fileStream << std::setw(4) << j << std::endl;
	fileStream.close();
}

b3d::tools::project::catalog::FileCatalog b3d::tools::project::catalog::FileCatalog::createOrLoadCatalogInDirectory(
	const std::filesystem::path& absoluteRootPath, const std::string& catalogFileName)
{
	if (!std::filesystem::exists(absoluteRootPath))
	{
		std::filesystem::create_directories(absoluteRootPath);
	}

	// Check if catalog already exists in dataRootPath. Parse catalog if it exists otherwise create a new one.
	const auto catalogFilePath = absoluteRootPath / catalogFileName;
	FileCatalog catalog{ absoluteRootPath, catalogFileName };
	if (std::filesystem::exists(catalogFilePath))
	{
		try
		{
			std::ifstream f(catalogFilePath);

			const auto data = nlohmann::json::parse(f);
			auto c = data.get<project::catalog::FileCatalog>();

			c.setRootPath(absoluteRootPath);
			c.setFileName(catalogFileName);

			c.removeInvalidMappings();
			catalog = c;
		}
		catch ([[maybe_unused]] nlohmann::json::type_error& e)
		{
			// std::cout << e.what();
			// [json.exception.type_error.304] cannot use at() with object
		}
		catch ([[maybe_unused]] nlohmann::json::parse_error& e)
		{
			// std::cout << e.what();
		}
		catch ([[maybe_unused]] nlohmann::json::exception& e)
		{
			// std::cout << e.what();
		}
	}

	catalog.writeCatalog();
	return catalog;
}
