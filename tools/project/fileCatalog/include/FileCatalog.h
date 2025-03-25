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
		FileCatalog(const std::filesystem::path& rootPath, const std::string& filename = DEFAULT_CATALOG_FILE_NAME);

		/// \brief Adds a file path to the catalog. Absolute path. The path must be inside the root path of the catalog
		auto addFilePathAbsolute(const std::filesystem::path& filePath, bool relativizePath = true) -> const std::string;

		/// \brief Adds a file path to the catalog. Path is relative to the root path of the catalog.
		auto addFilePathRelativeToRoot(const std::filesystem::path& filePath) -> const std::string;

		/// \brief Adds a file path with a given uuid to the catalog. Absolute path. The path must be inside the root path of the catalog.
		auto addFilePathAbsoluteWithUUID(const std::filesystem::path& filePath, const std::string& fileUUID) -> void;

		/// \brief Does the catalog contain a file with the given UUID?
		auto contains(const std::string& fileUUID) const -> bool;

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
		auto getRootPath() const -> const std::filesystem::path;

		/// Set rootPath. Required if Catalog was parsed from json
		auto setRootPath(const std::filesystem::path& rootPath) -> void;
		
		/// Set fileName. Required if Catalog was parsed from json
		auto setFileName(const  std::string& fileName) -> void;

		/// Write the catalog to a file in the root path of the catalog.
		auto writeCatalog() const -> void;

		///	\brief returns a Catalog from a existing catalog file in absoluteRootPath or creates a new one.
		///	\param absoluteRootPath The absolute path to the root folder of the catalog.
		static FileCatalog createOrLoadCatalogInDirectory(const std::filesystem::path& absoluteRootPath,
														  const std::string& catalogFileName = DEFAULT_CATALOG_FILE_NAME);

		inline static const std::string DEFAULT_CATALOG_FILE_SUFFIX = ".json";
		inline static const std::string DEFAULT_CATALOG_FILE_NAME = "catalog" + DEFAULT_CATALOG_FILE_SUFFIX;

	private:
		std::string b3dUuidFileCatalogVersion_{ "1.0" };

		// Absolute Path!
		std::filesystem::path rootPath_{ "/" };

		std::string fileName_{ DEFAULT_CATALOG_FILE_NAME };
	
		std::map<std::string, FileCatalogEntry> mappings_ {};

		#ifdef B3D_USE_NLOHMANN_JSON
				NLOHMANN_DEFINE_TYPE_INTRUSIVE(FileCatalog, b3dUuidFileCatalogVersion_, mappings_);
		#endif
	};
} // namespace b3d::tools::project::catalog
