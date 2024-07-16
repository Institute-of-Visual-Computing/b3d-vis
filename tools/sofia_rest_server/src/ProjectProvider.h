#pragma once

#include <filesystem>
#include <unordered_set>

#include "Catalog.h"
#include "Project.h"

namespace b3d::tools::projectexplorer
{
	class ProjectProvider
	{
	public:
		ProjectProvider(const std::filesystem::path& serverRootPath) : serverRootPath(serverRootPath)
		{
			generateGlobalFileCatalog();
			
			findProjectsInRootDirectory();
		}

		auto getProjects() -> const std::unordered_map<std::string, Project>&
		{
			return knownProjects;
		}

		auto getProject(const std::string& projectGUID) -> Project&
		{
			return knownProjects[projectGUID];
		}

		auto hasProject(const std::string& projectGUID) const -> bool
		{
			return knownProjects.contains(projectGUID);
		}

		auto getRootCatalog() -> Catalog&
		{
			return rootFileCatalog;
		}

		auto saveRootCatalog() -> bool;
		auto saveProject(const std::string& projectGUID) -> bool;

		auto getRootPath() -> const std::filesystem::path
		{
			return serverRootPath;
		}

	private:
		auto findProjectsInRootDirectory() -> void;
		auto findCatalogInRootDirectory() -> void;


		auto generateGlobalFileCatalog() -> void;

		std::filesystem::path serverRootPath;
		std::unordered_map<std::string, Project> knownProjects{};
		Catalog rootFileCatalog;
	};
} // namespace b3d::tools::projectexplorer
