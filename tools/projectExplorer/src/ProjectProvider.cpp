
#include <nlohmann/json.hpp>

#include <fstream>

#include "ProjectProvider.h"


#include "Catalog.h"


auto b3d::tools::projectexplorer::ProjectProvider::saveRootCatalog() -> bool
{
	const auto catalogFilePath = serverRootPath / "catalog.json";

	std::ofstream ofs(catalogFilePath, std::ofstream::trunc);
	nlohmann::json j = rootFileCatalog;
	ofs << std::setw(4) << j << std::endl;
	ofs.close();
	return true;
}

auto b3d::tools::projectexplorer::ProjectProvider::saveProject(const std::string& projectGUID) -> bool
{
	if (!knownProjects.contains(projectGUID))
	{
		return false;
	}

	auto& project = knownProjects.at(projectGUID);
	const auto projectFilePath = project.projectPathAbsolute / "project.json";

	std::ofstream ofs(projectFilePath, std::ofstream::trunc);
	nlohmann::json j = project;
	ofs << std::setw(4) << j << std::endl;
	ofs.close();
	return true;
}

auto b3d::tools::projectexplorer::ProjectProvider::findProjectsInRootDirectory() -> void
{
	for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(serverRootPath))
	{
		if (dirEntry.is_regular_file() && dirEntry.path().filename() == "project.json")
		{
			try
			{
				std::ifstream f(dirEntry.path());
				const auto data = nlohmann::json::parse(f);
				
				auto project = data.get<Project>();
				
				project.projectPathAbsolute = dirEntry.path().parent_path();
				knownProjects.insert(std::make_pair(project.projectUUID, project));
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
			catch(nlohmann::json::exception& e)
			{
				// std::cout << e.what();
			}
		}
	}
}

auto b3d::tools::projectexplorer::ProjectProvider::findCatalogInRootDirectory() -> void
{
	for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(serverRootPath))
	{
		if (dirEntry.is_regular_file() && dirEntry.path().extension() == ".json")
		{
			try
			{

				std::ifstream f(dirEntry.path());
				const auto data = nlohmann::json::parse(f);
				const auto project = data.get<Catalog>();
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
	}
}

auto b3d::tools::projectexplorer::ProjectProvider::generateGlobalFileCatalog() -> void
{
	Catalog c;
	c.serverRootPath = serverRootPath;
	const auto dataPath = serverRootPath / "data";
	const auto catalogFilePath = serverRootPath / "catalog.json";
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
			c = data.get<Catalog>();
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


	c.serverRootPath = serverRootPath;

	c.removeInvalidMappings();
	c.relativeMappings();

	for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(serverRootPath / "data"))
	{
		if (dirEntry.is_regular_file() && dirEntry.path().extension() == ".fits")
		{
			c.addFilePath(dirEntry.path());
		}
	}

	rootFileCatalog = c;
	saveRootCatalog();
}
