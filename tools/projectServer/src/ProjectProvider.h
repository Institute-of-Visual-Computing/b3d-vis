#pragma once

#include <filesystem>

#include "FileCatalog.h"
#include "Project.h"

namespace b3d::tools::projectServer
{
	class ProjectProvider
	{
	public:
		ProjectProvider(const std::filesystem::path& rootPath) : rootPath_(rootPath), catalog_(rootPath)
		{
			generateCatalog();
			findProjects();
			flagInvalidFilesInProjects();
		}

		auto getProjects() -> const std::map<std::string, b3d::tools::project::Project>&
		{
			return knownProjects_;
		}

		auto getProject(const std::string& projectUUID) -> b3d::tools::project::Project&
		{
			return knownProjects_[projectUUID];
		}

		auto hasProject(const std::string& projectUUID) const -> bool
		{
			return knownProjects_.contains(projectUUID);
		}

		auto getCatalog() -> project::catalog::FileCatalog&
		{
			return catalog_;
		}

		auto saveRootCatalog() -> bool;
		auto saveProject(const std::string& projectUUID) -> bool;

		auto getRootPath() -> std::filesystem::path
		{
			return rootPath_;
		}

		auto getProjectsPath() const  -> std::filesystem::path
		{
			return projectsPath_;
		}

		auto getProjectsPathAbsolute() const  -> std::filesystem::path
		{
			return rootPath_ / projectsPath_;
		}

		auto getDataPath() const -> std::filesystem::path
		{
			return dataPath_;
		}

		auto getDataPathAbsolute() const -> std::filesystem::path
		{
			return rootPath_ / dataPath_;
		}

	private:
		auto findProjects() -> void;

		auto generateCatalog() -> void;

		auto flagInvalidFilesInProjects() -> void;

		// Absolute Path!
		std::filesystem::path rootPath_ {"/" };
		// relative to root path
		std::filesystem::path dataPath_{ "data" };
		// relative to root path
		std::filesystem::path projectsPath_{ "projects" };

		std::map<std::string, b3d::tools::project::Project> knownProjects_{};
		project::catalog::FileCatalog catalog_;
	};
} // b3d::tools::projectServer
