#include "Catalog.h"

#include "uuid.h"
#include "nlohmann/json.hpp"

#include "Common.h"

// Absolute Path!
auto b3d::tools::projectexplorer::Catalog::addFilePath(std::filesystem::path filePath, const bool cached)
-> const std::string
{
		auto fileUUID = to_string(gNameGenerator(filePath.string()));
	CatalogEntry ce{ filePath.lexically_relative(serverRootPath), filePath.filename() };
		if (cached)
		{
			cachedMappings.insert(std::make_pair(fileUUID, ce));
		}
		else
		{
			mappings.insert(std::make_pair(fileUUID, ce));
		}
		return fileUUID;
}

auto b3d::tools::projectexplorer::Catalog::getExternalJsonRepresentation() const -> std::string
{
	std::vector<std::string_view> keys;
	for (const auto& key : mappings | std::views::keys | std::views::all)
	{
		keys.emplace_back(key);
	}
	for (const auto& key : cachedMappings | std::views::keys | std::views::all)
	{
		keys.emplace_back(key);
	}
	return nlohmann::json(keys).dump();
}
