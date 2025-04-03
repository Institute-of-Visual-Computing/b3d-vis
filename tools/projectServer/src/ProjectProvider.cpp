#include "ProjectProvider.h"

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include <FileCatalog.h>
#include <FitsTools.h>


auto b3d::tools::projectServer::ProjectProvider::addExistingProject(const std::string& uuid) -> const bool
{
	bool foundProject = false;
	std::string projectUUID;
	
	try
	{
		std::ifstream f(getProjectsPathAbsolute() / uuid / "project.json");
		const auto data = nlohmann::json::parse(f);
		f.close();
		auto project = data.get<project::Project>();

		// Read Props for fits
		// project.fitsOriginProperties.unit

		projectUUID = project.projectUUID;
		knownProjects_.insert(std::make_pair(project.projectUUID, project));
		saveProject(project.projectUUID);
		foundProject = true;
	}
	catch (nlohmann::json::type_error& e)
	{
		std::cout << e.what();
		// [json.exception.type_error.304] cannot use at() with object
	}
	catch (nlohmann::json::parse_error& e)
	{
		std::cout << e.what();
	}
	catch (nlohmann::json::exception& e)
	{
		std::cout << e.what();
	}
	
	if (foundProject)
	{
		auto catalog = project::catalog::FileCatalog::createOrLoadCatalogInDirectory(
			getProjectsPathAbsolute() / uuid);

		knownFileCatalogs_.insert(std::make_pair(projectUUID, catalog));

	}
	return foundProject;
}

auto b3d::tools::projectServer::ProjectProvider::getCatalog(
	const std::string& projectUUID) -> project::catalog::FileCatalog&
{
	if (!knownFileCatalogs_.contains(projectUUID))
	{
		auto projectFileCatalog = project::catalog::FileCatalog::createOrLoadCatalogInDirectory(
			knownProjects_[projectUUID].projectPathAbsolute);
		knownFileCatalogs_.insert(std::make_pair(projectUUID, projectFileCatalog));
	}
	return knownFileCatalogs_[projectUUID];
}

auto b3d::tools::projectServer::ProjectProvider::saveProject(const std::string& projectUUID) -> bool
{
	if (!knownProjects_.contains(projectUUID))
	{
		return false;
	}

	auto& project = knownProjects_.at(projectUUID);
	const auto projectFilePath = project.projectPathAbsolute / "project.json";

	std::ofstream ofs(projectFilePath, std::ofstream::trunc);
	nlohmann::json j = project;
	ofs << std::setw(4) << j << std::endl;
	ofs.close();
	return true;
}

auto b3d::tools::projectServer::ProjectProvider::removeProject(const std::string& projectUUID) -> bool
{
	auto projectToRemove = knownProjects_.at(projectUUID);
	auto projectRemoved = knownProjects_.erase(projectUUID) > 0;

	if (projectRemoved)
	{
		std::filesystem::remove_all(projectToRemove.projectPathAbsolute);
	}
	return true;
}

auto b3d::tools::projectServer::ProjectProvider::findProjects() -> void
{
	for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(getProjectsPathAbsolute()))
	{
		bool foundProject = false;
		std::string projectUUID;
		if (dirEntry.is_regular_file() && dirEntry.path().filename() == "project.json")
		{
			try
			{
				std::ifstream f(dirEntry.path());
				const auto data = nlohmann::json::parse(f);
				f.close();
				auto project = data.get<project::Project>();

				// Read Props for fits
				// project.fitsOriginProperties.unit

				project.projectPathAbsolute = dirEntry.path().parent_path();
				projectUUID = project.projectUUID;
				knownProjects_.insert(std::make_pair(project.projectUUID, project));
				saveProject(project.projectUUID);
				foundProject = true;

			}
			catch (nlohmann::json::type_error& e)
			{
				std::cout << e.what();
				// [json.exception.type_error.304] cannot use at() with object
			}
			catch (nlohmann::json::parse_error& e)
			{
				std::cout << e.what();
			}
			catch(nlohmann::json::exception& e)
			{
				std::cout << e.what();
			}
		}
		if (foundProject)
		{
			auto catalog = project::catalog::FileCatalog::createOrLoadCatalogInDirectory(dirEntry.path().parent_path());

			knownFileCatalogs_.insert(std::make_pair(projectUUID, catalog));
		}
	}
}

auto b3d::tools::projectServer::ProjectProvider::clearMissingRequests() -> void
{
	for (auto& project : knownProjects_)
	{
		auto& requests = project.second.requests;
		std::erase_if(requests,
		              [&](const project::Request& request) {
			              return !std::filesystem::exists(project.second.projectPathAbsolute /
				              "requests" / request.uuid);
		              });
	}
}

auto b3d::tools::projectServer::ProjectProvider::flagInvalidFilesInProjects() -> void
{
	for (auto& project : knownProjects_) {

		auto&& projectCatalog = getCatalog(project.first);
		for (auto& request : project.second.requests)
		{
			if (request.result.sofiaResult.wasSuccess() || request.result.sofiaResult.resultFile.empty())
			{
				const auto filePath = projectCatalog.getFilePathAbsolute(request.result.sofiaResult.resultFile);
				if (filePath.empty())
				{
					request.result.sofiaResult.fileAvailable = false;
				}
			}
			else
			{
				request.result.sofiaResult.fileAvailable = false;
			}

			if (request.result.nanoResult.wasSuccess() || request.result.nanoResult.resultFile.empty())
			{
				const auto filePath = projectCatalog.getFilePathAbsolute(request.result.nanoResult.resultFile);
				if (filePath.empty())
				{
					request.result.nanoResult.fileAvailable = false;
				}
			}
			else
			{
				request.result.nanoResult.fileAvailable = false;
			}

		}
		saveProject(project.first);
	}
}
