#include <ranges>

#include <uuid.h>

#include "FileCatalog.h"

static uuids::uuid_name_generator gNameGenerator{ uuids::uuid::from_string("123456789-abcdef-123456789-abcdef-12").value() };

b3d::tools::project::catalog::FileCatalog::FileCatalog(const std::filesystem::path& rootPath) : rootPath_(rootPath)
{
}

b3d::tools::project::catalog::FileCatalog::FileCatalog(const std::filesystem::path& rootPath,
	const std::filesystem::path& dataPath, const std::filesystem::path& projectsPath) : rootPath_(rootPath), dataPath_(dataPath), projectsPath_(projectsPath)
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
