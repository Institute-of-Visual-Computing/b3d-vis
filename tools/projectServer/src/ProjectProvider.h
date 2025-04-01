#pragma once

#include <filesystem>

#include "FileCatalog.h"
#include "Project.h"

namespace b3d::tools::projectServer
{
	class ProjectProvider
	{
	public:
		ProjectProvider(const std::filesystem::path& rootPath) : rootPath_(rootPath)
		{
			findProjects();
			clearMissingRequests();
			flagInvalidFilesInProjects();
		}

		auto addExistingProject(const std::string &uuid) -> const bool;

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

		auto getCatalog(const std::string& projectUUID) -> project::catalog::FileCatalog&;

		auto getAllCatalogs() -> const std::map<std::string, b3d::tools::project::catalog::FileCatalog>&
		{
			return knownFileCatalogs_;
		}

		auto saveProject(const std::string& projectUUID) -> bool;

		auto removeProject(const std::string& projectUUID) -> bool;

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


	private:
		auto findProjects() -> void;

		auto clearMissingRequests() -> void;

		// auto generateCatalog() -> void;

		auto flagInvalidFilesInProjects() -> void;

		// Absolute Path!
		std::filesystem::path rootPath_ {"/" };

		// relative to root path
		std::filesystem::path projectsPath_{ "projects" };

		std::map<std::string, b3d::tools::project::Project> knownProjects_{};

		std::map<std::string, b3d::tools::project::catalog::FileCatalog> knownFileCatalogs_{};

	};
} // b3d::tools::projectServer
