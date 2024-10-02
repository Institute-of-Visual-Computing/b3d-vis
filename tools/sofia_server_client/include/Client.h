#pragma once
#include <future>
#include <string>

#include "Project.h"

namespace b3d::tools::server_client
{
	class Client
	{
	public:
		Client(std::string host = "localhost", int port = 5051) : host_(host), port_(port){}

		// auto getProjectsAsync() -> std::future<std::unordered_map<std::string, b3d::tools::projectexplorer::Project>>;

		auto getProjects() -> std::vector<projectexplorer::Project>;

		auto getFile(std::string fileGUID) -> std::filesystem::path;
		auto getFileAsync(std::string fileGUID) -> std::future<std::filesystem::path>;

	private:
		std::string host_;
		int port_;
	};
}
