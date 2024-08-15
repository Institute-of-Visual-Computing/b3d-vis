#pragma once

#include <future>

#include "Project.h"

#ifdef B3D_USE_NLOHMANN_JSON
	#include <nlohmann/json.hpp>
#endif

namespace b3d::tools::project
{
	enum class ServerStatusState
	{
		ok,
		unreachable,
		unknown,
		testing
	};

	/// This struct is used to store the connection information for the server
	struct ServerConnectionDescription
	{
		/// The port the server is running on
		std::string port{ "5051" };
		/// The IP or Hostname of the server
		std::string ipHost{ "localhost" };
		/// The name of the server
		std::string name{ "localhost" };

		#ifdef B3D_USE_NLOHMANN_JSON
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(ServerConnectionDescription, port, ipHost, name);
		#endif
	};

	/// This class is used to connect to the server and get information about the projects and requests
	class ServerClient
	{
	public:
		ServerClient() = default;
		ServerClient(ServerConnectionDescription serverConnectionDescription);

		auto setNewConnectionInfo(ServerConnectionDescription serverConnectionDescription) -> void;
		auto getConnectionInfo() -> const ServerConnectionDescription&;

		auto getLastServerStatusState() -> ServerStatusState;
		auto getServerStatusStateAsync() -> std::future<ServerStatusState>;

		auto updateServerStatusState(float deltaTimeSeconds) -> void;
		auto forceUpdateServerStatusState() -> void;

		auto getProjects() -> std::vector<Project>;
		auto getProject(const std::string& projectUUID) -> Project;
		auto getRequests(const std::string& projectUUID) -> std::vector<Request>;
		auto getRequest(const std::string& projectUUID, const std::string& requestUUID) -> std::vector<Request>;

		static constexpr float heartbeatIntervalSeconds = 5.0f;

	private:
		auto getServerStatusState(ServerConnectionDescription connectionDescription) -> ServerStatusState;

		// IP or Hostname
		ServerConnectionDescription serverConnectionDescription_
		{
			.port = "5051",
			.ipHost = "localhost",
			.name = "localhost"
		};

		ServerStatusState lastServerStatusState_{ ServerStatusState::unknown };

		bool lastHeartbeatDone_ = true ;
		float secondsSinceLastHeartbeat_ = heartbeatIntervalSeconds;
		std::future<ServerStatusState> heartbeatFuture_;
	};
}
