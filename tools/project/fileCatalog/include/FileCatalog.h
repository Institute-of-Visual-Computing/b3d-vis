#pragma once

#include <filesystem>
#include <map>
#include <string>

#ifdef B3D_USE_NLOHMANN_JSON
	#include <nlohmann/json.hpp>
#endif

namespace b3d::tools::project::catalog
{
	/// \brief A FileCatalogEntry contains the file path, the file name and additional file information.
	struct FileCatalogEntry
	{
		std::filesystem::path filePath;
		std::filesystem::path fileName;
		std::map<std::string, std::string> fileInfos;

		#ifdef B3D_USE_NLOHMANN_JSON
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(FileCatalogEntry, filePath, fileName, fileInfos);
		#endif
	};

	//// \brief The FileCatalog class is used to store the mapping between a file UUID and the file path.
	class FileCatalog
	{
	public:
		FileCatalog() = default;
		FileCatalog(const std::filesystem::path& rootPath);
		FileCatalog(const std::filesystem::path& rootPath, const std::filesystem::path& dataPath,
				const std::filesystem::path& projectsPath);

		/// \brief Adds a file path to the catalog. Absolute path.
		auto addFilePathAbsolute(const std::filesystem::path& filePath) -> const std::string;

		/// \brief Adds a file path to the catalog. Path is relative to the root path of the catalog.
		auto addFilePathRelativeToRoot(const std::filesystem::path& filePath) -> const std::string;

		/// \brief Get the file path relative to the root path of the catalog for a given file UUID.
		auto getFilePathRelativeToRoot(const std::string& fileUUID) const -> const std::filesystem::path;

		/// \brief Get the absolute file path for a given file UUID.
		auto getFilePathAbsolute(const std::string& fileUUID) const -> const std::filesystem::path;

		/// \brief Get all Mappings.
		auto getMappings() const -> const std::map<std::string, FileCatalogEntry>&;

		/// \brief Remove a Mapping.
		auto removeMapping(const std::string& fileUUID) -> void;

		/// \brief Modify all file paths in the catalog to be relative to the root path of the catalog.
		auto relativeMappings() -> void;

		/// \brief Remove non existing mappings.
		auto removeInvalidMappings() -> void;

		/// Absolute Path!
		auto setRootPath(const std::filesystem::path& rootPath) -> void;

		/// Absolute Path!
		auto getRootPath() const -> const std::filesystem::path;

		/// \brief Get the absolute path to the data folder.
		auto getDataPathAbsolute() const -> const std::filesystem::path;

	private:
		std::string b3dUuidFileCatalogVersion_{ "1.0" };

		// Absolute Path!
		std::filesystem::path rootPath_{ "/" };

		// relative to root path
		std::filesystem::path dataPath_{ "data" };

		// relative to root path
		std::filesystem::path projectsPath_ { "projects" };

		std::map<std::string, FileCatalogEntry> mappings_ {};

		#ifdef B3D_USE_NLOHMANN_JSON
				NLOHMANN_DEFINE_TYPE_INTRUSIVE(FileCatalog, b3dUuidFileCatalogVersion_, mappings_, dataPath_, projectsPath_);
		#endif
	};
} // namespace b3d::tools::project::catalog
