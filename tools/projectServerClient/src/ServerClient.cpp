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

auto ServerClient::setNewConnectionInfo(
	ServerConnectionDescription serverConnectionDescription) -> void
{
	serverConnectionDescription_ = std::move(serverConnectionDescription);
	lastServerStatusState_ = ServerStatusState::unknown;
	lastHeartbeatDone_ = true;
	updateServerStatusState(100.0f);
}

auto ServerClient::getConnectionInfo() -> const ServerConnectionDescription&
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

auto ServerClient::getServerStatusState(const ServerConnectionDescription& connectionDescription)
	-> ServerStatusState
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

auto ServerClient::getProjects(const ServerConnectionDescription& connectionDescription)
	-> std::optional<std::vector<Project>>
{
	nlohmann::json ja{""};
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
