#include "Client.h"

#include "httplib.h"

auto b3d::tools::server_client::Client::getProjects() -> std::vector<projectexplorer::Project>
{
	httplib::Client cli(host_, port_);
	if (auto res = cli.Get("/projects"))
	{
		if (res->status == httplib::StatusCode::OK_200)
		{
			try
			{
				const auto data = nlohmann::json::parse(res->body);
				const auto projects = data.get<std::vector<projectexplorer::Project>>();
				return projects;
			}
			catch (nlohmann::json::type_error& e)
			{
				std::cout << e.what();
				// [json.exception.type_error.304] cannot use at() with object
			}
		}
	}
	else
	{
		auto err = res.error();
		std::cout << "HTTP error: " << httplib::to_string(err) << std::endl;
	}
	return {};
}

auto b3d::tools::server_client::Client::getFile(std::string fileGUID) -> std::filesystem::path
{
	httplib::Client cli(host_, port_);
	const auto filePath = std::filesystem::path{ std::format("{}.nvdb", fileGUID) };
	std::ofstream output(std::format("{}.nvdb", fileGUID), std::ios::binary);
	auto res = cli.Get(
		std::format("/file/{}",fileGUID),
		[](const httplib::Response &response)
		{
			std::cout << "Reponse is " << response.status << "\n";
			return true;
		},
		[&output](const char* data, size_t data_length)
		{
			std::cout << "Receiving " <<  data_length << " bytes\n";
			output.write(data, data_length);
			return true;
		}, [](size_t len, size_t total)
		{
			// std::cout << "Progres: " << len << " from " << total << "\n";
			return true;
		}
	);

	output.close();
	if (res.error() != httplib::Error::Success)
	{
		return {};
	}
	
	return filePath;
}

auto b3d::tools::server_client::Client::getFileAsync(std::string fileGUID) -> std::future<std::filesystem::path>
{
	auto filePathFuture = std::async(std::launch::async, &Client::getFile, this, fileGUID);
	return filePathFuture;
}
