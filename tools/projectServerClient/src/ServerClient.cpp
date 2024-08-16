#include <httplib.h>

#include "ServerClient.h"

b3d::tools::project::ServerClient::ServerClient(ServerConnectionDescription serverConnectionDescription)
	: serverConnectionDescription_(std::move(serverConnectionDescription))
{
}

auto b3d::tools::project::ServerClient::setNewConnectionInfo(
	ServerConnectionDescription serverConnectionDescription) -> void
{
	serverConnectionDescription_ = std::move(serverConnectionDescription);
	lastServerStatusState_ = ServerStatusState::unknown;
	lastHeartbeatDone_ = true;
	updateServerStatusState(100.0f);
}

auto b3d::tools::project::ServerClient::getConnectionInfo() -> const ServerConnectionDescription&
{
	return serverConnectionDescription_;
}

auto b3d::tools::project::ServerClient::getLastServerStatusState() -> ServerStatusState
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

auto b3d::tools::project::ServerClient::getServerStatusStateAsync() -> std::future<ServerStatusState>
{ 
	return std::async(std::launch::async, [this]() { return getServerStatusState(serverConnectionDescription_); });
}

auto b3d::tools::project::ServerClient::updateServerStatusState(float deltaTimeSeconds) -> void
{
	// Start heartbeat heartBeatIntervalSeconds_ Seconds after last heartbeat returned.
	if (lastHeartbeatDone_)
	{
		secondsSinceLastHeartbeat_ += deltaTimeSeconds;
	}

	if (secondsSinceLastHeartbeat_ >= heartbeatIntervalSeconds && lastHeartbeatDone_)
	{
		lastHeartbeatDone_ = false;
		lastServerStatusState_ =
			lastServerStatusState_ == ServerStatusState::ok ? ServerStatusState::ok : ServerStatusState::testing;
		heartbeatFuture_ = getServerStatusStateAsync();
	}
}

auto b3d::tools::project::ServerClient::forceUpdateServerStatusState() -> void
{
	if (heartbeatFuture_.valid())
	{
		return;
	}
	updateServerStatusState(heartbeatIntervalSeconds + 1.0f);
}

auto b3d::tools::project::ServerClient::getServerStatusState(ServerConnectionDescription connectionDescription)
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
