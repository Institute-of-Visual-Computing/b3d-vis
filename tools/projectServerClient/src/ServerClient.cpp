#include <filesystem>
#include <future>
#include <httplib.h>
#include <string>
#include <vector>

#include <args.hxx>
#include <nlohmann/json.hpp>

#include "Project.h"

#include "ServerClient.h"

using namespace b3d::tools::project;

ServerClient::ServerClient(ServerConnectionDescription serverConnectionDescription)
	: serverConnectionDescription_(std::move(serverConnectionDescription))
{
}

auto ServerClient::setNewConnectionInfo(ServerConnectionDescription serverConnectionDescription) -> void
{
	serverConnectionDescription_ = std::move(serverConnectionDescription);
	lastServerStatusState_ = ServerStatusState::unknown;
	lastHeartbeatDone_ = true;
	updateServerStatusState(100.0f);
}

auto ServerClient::getConnectionInfo() const -> const ServerConnectionDescription&
{
	return serverConnectionDescription_;
}

auto ServerClient::getLastServerStatusState() -> ServerStatusState
{
	if (heartbeatFuture_.valid())
	{
		if (heartbeatFuture_.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
		{
			lastServerStatusState_ = heartbeatFuture_.get();
			lastHeartbeatDone_ = true;
			secondsSinceLastHeartbeat_ = 0.0f;
		}
	}
	return lastServerStatusState_;
}

auto ServerClient::getServerStatusStateAsync() const -> std::future<ServerStatusState>
{
	return std::async(std::launch::async, [this]() { return getServerStatusState(serverConnectionDescription_); });
}

auto ServerClient::updateServerStatusState(float deltaTimeSeconds) -> void
{
	// Start heartbeat heartBeatIntervalSeconds_ Seconds after last heartbeat returned.
	if (lastHeartbeatDone_)
	{
		secondsSinceLastHeartbeat_ += deltaTimeSeconds;
	}

	if (secondsSinceLastHeartbeat_ >= ServerClient::heartbeatIntervalSeconds && lastHeartbeatDone_)
	{
		lastHeartbeatDone_ = false;
		lastServerStatusState_ =
			lastServerStatusState_ == ServerStatusState::ok ? ServerStatusState::ok : ServerStatusState::testing;
		heartbeatFuture_ = getServerStatusStateAsync();
	}
}

auto ServerClient::forceUpdateServerStatusState() -> void
{
	if (heartbeatFuture_.valid())
	{
		return;
	}
	updateServerStatusState(ServerClient::heartbeatIntervalSeconds + 1.0f);
}

auto ServerClient::getProjectsAsync() const -> std::future<std::optional<std::vector<Project>>>
{
	return std::async(std::launch::async, [this]() { return getProjects(serverConnectionDescription_); });
}

auto ServerClient::downloadFileAsync(const std::string& fileUUID, const std::filesystem::path &targetDirectoryPath) const
	-> std::future<std::filesystem::path>
{
	return std::async(std::launch::async,[this, fileUUID, targetDirectoryPath]() { return downloadFile(serverConnectionDescription_, fileUUID, targetDirectoryPath); });
}

auto ServerClient::startSearchAsync(const std::string& projectUUID, const sofia::SofiaParams& params,
	bool force) -> std::future<std::string>
{
	return std::async(std::launch::async,
					  [this, projectUUID, params, force]()
					  { return startSearch(serverConnectionDescription_, projectUUID, params, force); });
}

auto ServerClient::getServerStatusState(const ServerConnectionDescription connectionDescription) -> ServerStatusState
{
	auto statusState = ServerStatusState::testing;
	httplib::Client client(connectionDescription.ipHost, std::stoi(connectionDescription.port));
	auto res = client.Get("/status");
	if (!res)
	{
		statusState = ServerStatusState::unreachable;
	}
	else if (res.error() != httplib::Error::Success)
	{
		statusState = ServerStatusState::unreachable;
	}
	else if (res->status == 200)
	{
		statusState = ServerStatusState::ok;
	}
	else
	{
		statusState = ServerStatusState::unknown;
	}
	return statusState;
}

auto ServerClient::getProjects(const ServerConnectionDescription connectionDescription)
	-> std::optional<std::vector<Project>>
{
	httplib::Client client(connectionDescription.ipHost, std::stoi(connectionDescription.port));
	auto res = client.Get("/projects");

	if (!res || res.error() != httplib::Error::Success)
	{
		return std::nullopt;
	}
	if (res->status == 200)
	{
		const auto jsonObj = nlohmann::json::parse(res->body);
		try
		{
			return jsonObj.get<std::vector<Project>>();
		}
		catch (const nlohmann::json::exception& e)
		{
			// Todo: Log error
			return std::nullopt;
		}
	}
	return std::nullopt;
}

auto ServerClient::downloadFile(const ServerConnectionDescription connectionDescription, const std::string fileUUID,
								const std::filesystem::path targetDirectoryPath) -> std::filesystem::path
{
	httplib::Client client(connectionDescription.ipHost, std::stoi(connectionDescription.port));

	const auto targetFilePath = targetDirectoryPath / (fileUUID + ".tmp");

	std::ofstream myfile;
	myfile.open(targetFilePath, std::ios::out | std::ios::binary | std::ios::trunc);

	auto res = client.Get(std::format("/file/{}", fileUUID),
						  [&](const char* data, size_t data_length)
						  {
							  myfile.write(data, data_length);
							  return true;
						  });
	myfile.close();
	if (!res || res.error() != httplib::Error::Success || res->status != 200)
	{
		std::filesystem::remove(targetFilePath);
		return {};
	}

	const auto extension = res->get_header_value("Content-Type").substr(12 /* application/ */);
	const auto finalFilePath = targetDirectoryPath / (fileUUID + "." + extension); 
	std::filesystem::rename(targetFilePath,finalFilePath);
	return finalFilePath;
}

auto ServerClient::startSearch(ServerConnectionDescription connectionDescription, const std::string& projectUUID,
							   const sofia::SofiaParams& params,
	bool force) -> std::string
{
	httplib::Client client(connectionDescription.ipHost, std::stoi(connectionDescription.port));
	nlohmann::json requestJSON;

	requestJSON["projectUUID"] = projectUUID;
	requestJSON["force"] = force;
	requestJSON["sofia_params"] = params;
	const auto requestString = requestJSON.dump();

	auto res = client.Post("/startSearch", requestString.c_str(), "application/json");

	if (!res || res.error() != httplib::Error::Success)
	{
		return "";
	}
	if (res->status == 200)
	{
		const auto jsonObj = nlohmann::json::parse(res->body);
		if (jsonObj.contains("requestUUID"))
		{
			return jsonObj["requestUUID"];
		}
	}
	return "";
}
