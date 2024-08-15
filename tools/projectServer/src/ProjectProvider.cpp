
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include "FileCatalog.h"
#include "FitsTools.h"

#include "ProjectProvider.h"

auto b3d::tools::projectServer::ProjectProvider::saveRootCatalog() -> bool
{
	const auto catalogFilePath = rootPath_ / "catalog.json";

	std::ofstream ofs(catalogFilePath, std::ofstream::trunc);

	nlohmann::json j = catalog_;
	ofs << std::setw(4) << j << std::endl;
	ofs.close();
	return true;
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

auto b3d::tools::projectServer::ProjectProvider::findProjects() -> void
{
	for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(getProjectsPathAbsolute()))
	{
		if (dirEntry.is_regular_file() && dirEntry.path().filename() == "project.json")
		{
			try
			{
				std::ifstream f(dirEntry.path());
				const auto data = nlohmann::json::parse(f);
				f.close();
				auto project = data.get<project::Project>();

				auto fitsRootPath = catalog_.getFilePathAbsolute(project.fitsOriginUUID);
				project.fitsOriginProperties = b3d::tools::fits::getFitsProperties(fitsRootPath);

				// Read Props for fits
				// project.fitsOriginProperties.unit
								
				project.projectPathAbsolute = dirEntry.path().parent_path();
				knownProjects_.insert(std::make_pair(project.projectUUID, project));
				saveProject(project.projectUUID);
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
	}
}

auto b3d::tools::projectServer::ProjectProvider::generateCatalog() -> void
{
	project::catalog::FileCatalog c;
	c.setRootPath(rootPath_);

	const auto dataPath = getDataPathAbsolute();
	const auto catalogFilePath = rootPath_ / "catalog.json";

	if (!exists(dataPath))
	{
		std::filesystem::create_directories(dataPath);
	}
	if (exists(catalogFilePath))
	{
		try
		{
			std::ifstream f(catalogFilePath);

			const auto data = nlohmann::json::parse(f);
			c = data.get<project::catalog::FileCatalog>();
		}
		catch (nlohmann::json::type_error& e)
		{
			// std::cout << e.what();
			// [json.exception.type_error.304] cannot use at() with object
		}
		catch (nlohmann::json::parse_error& e)
		{
			// std::cout << e.what();
		}
		catch (nlohmann::json::exception& e)
		{
			// std::cout << e.what();
		}
	}

	c.setRootPath(rootPath_);
	c.removeInvalidMappings();
	c.relativeMappings();

	for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(getDataPathAbsolute()))
	{
		if (dirEntry.is_regular_file() &&
			(dirEntry.path().extension() == ".fits" || dirEntry.path().extension() == ".nvdb"))
		{
			c.addFilePathAbsolute(dirEntry.path());
		}
	}

	for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(getProjectsPathAbsolute()))
	{
		if (dirEntry.is_regular_file() &&
			(dirEntry.path().extension() == ".fits" || dirEntry.path().extension() == ".nvdb"))
		{
			c.addFilePathAbsolute(dirEntry.path());
		}
	}

	catalog_ = c;
	saveRootCatalog();
}
