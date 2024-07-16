#pragma once

#include <filesystem>
#include <map>
#include <string>

#include <unordered_map>

#include "nlohmann/json.hpp"

namespace b3d::tools::projectexplorer
{
	struct CatalogEntry
	{
		std::filesystem::path filePath;
		std::filesystem::path fileName;
		std::map<std::string, std::string> fileInfos;


		NLOHMANN_DEFINE_TYPE_INTRUSIVE(CatalogEntry, filePath, fileName, fileInfos);
	};

	class Catalog
	{
	public:
		auto addFilePath(std::filesystem::path filePath, const bool cached = false) -> const std::string;

		auto getFilePathRelativeToRoot(const std::string& fileGUID) const -> const std::filesystem::path
		{
			if (mappings.contains(fileGUID))
			{
				return mappings.at(fileGUID).filePath;
			}
			if (cachedMappings.contains(fileGUID))
			{
				return cachedMappings.at(fileGUID).filePath;
			}
			return {};
		}

		auto getFilePathAbsolute(const std::string& fileGUID) const -> const std::filesystem::path
		{
			if (mappings.contains(fileGUID))
			{
				return serverRootPath / mappings.at(fileGUID).filePath;
			}
			if (cachedMappings.contains(fileGUID))
			{
				return serverRootPath / cachedMappings.at(fileGUID).filePath;
			}
			return {};
		}

		auto getMappings() const -> const std::unordered_map<std::string, CatalogEntry>&
		{
			return mappings;
		}

		auto getCachedMappings() const -> const std::unordered_map<std::string, CatalogEntry>&
		{
			return cachedMappings;
		}

		auto removeMapping(const std::string &fileGUID) -> void
		{
			mappings.erase(fileGUID);
			cachedMappings.erase(fileGUID);
		}

		auto relativeMappings() -> void
		{
			for (auto& entry : mappings | std::views::values | std::views::all)
			{
				if (entry.filePath.is_absolute())
				{
					entry.filePath = entry.filePath.lexically_relative(serverRootPath);
				}
			}

			for (auto& entry : cachedMappings | std::views::values | std::views::all)
			{
				if (entry.filePath.is_absolute())
				{
					entry.filePath = entry.filePath.lexically_relative(serverRootPath);
				}
			}
		}

		auto removeInvalidMappings() -> void
		{
			std::vector<std::string> identifiersToRemove{};

			for (const auto& kv : mappings)
			{
				if (!exists(serverRootPath / kv.second.filePath))
				{
					identifiersToRemove.push_back(kv.first);
				}
			}
			for (const auto& kv : cachedMappings)
			{
				if (!exists(serverRootPath / kv.second.filePath))
				{
					identifiersToRemove.push_back(kv.first);
				}
			}

			for (const auto& guid : identifiersToRemove)
			{
				removeMapping(guid);
			}
		}

		auto getExternalJsonRepresentation() const -> std::string;
		
		std::string b3dGuidFileCatalogVersion{ "1.0.0" };
		std::unordered_map<std::string, CatalogEntry> mappings{};
		std::unordered_map<std::string, CatalogEntry> cachedMappings{};

		std::filesystem::path serverRootPath;
		
		NLOHMANN_DEFINE_TYPE_INTRUSIVE(Catalog, b3dGuidFileCatalogVersion, mappings, cachedMappings);
	};

} // namespace b3d::tools::projectexplorer
