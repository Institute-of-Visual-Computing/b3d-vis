#pragma once

#include <atomic>
#include <future>

#include <Project.h>

#ifdef B3D_USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace b3d::tools::project
{
	enum class ServerHealthState
	{
		ok,
		unreachable,
		unknown,
		testing
	};

	enum class ServerBusyState
	{
		idle,
		processing,
		unknown
	};

	struct ServerState
	{
		ServerHealthState health{ ServerHealthState::unknown };
		ServerBusyState busyState{ ServerBusyState::unknown };
	};

	enum class UploadState
	{
		unknown,
		ok,
		fileInvalid,
		uploadFaild
	};

	struct UploadResult
	{
		UploadState state;
		std::string projectName;
	};

	struct UploadFeedback
	{
		std::atomic<int> progress;
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

		/// \brief Replace the current connection information with the new one
		/// \param serverConnectionDescription The new connection information
		auto setNewConnectionInfo(ServerConnectionDescription serverConnectionDescription) -> void;

		/// \brief Get the current connection information
		auto getConnectionInfo() const -> const ServerConnectionDescription&;

		/// \brief Last known server status
		auto getLastServerStatusState() -> ServerState;

		/// \brief Ask the client to request the server status.
		auto getServerStatusStateAsync() const -> std::future<ServerState>;

		/// \brief Use the game loop to update the server status state in a heartbeat fashion
		/// \param deltaTimeSeconds Time in seconds since the last call
		auto updateServerStatusState(float deltaTimeSeconds) -> void;

		/// \brief Force the client to request a server status update
		auto forceUpdateServerStatusState() -> void;

		/// \brief Get the projects from the server
		auto getProjectsAsync() const -> std::future<std::optional<std::vector<Project>>>;

		auto downloadFileAsync(const std::string& fileUUID, const std::filesystem::path& targetDirectoryPath) const
			-> std::future<std::filesystem::path>;

		auto uploadFileAsync(const std::filesystem::path& sourceFile, const std::string& projectName,
							 UploadFeedback& uploadFeedback) const -> std::future<UploadResult>;

		auto deleteProjectAsync(const std::string projectUUID) const -> std::future<void>;

		auto isServerBusy() const -> std::future<bool>;

		/// \brief Start a search on the server
		/// \param projectUUID
		/// \param params
		/// \param force Force server to run request again
		/// \return UUID of the request, if started
		auto startSearchAsync(const std::string& projectUUID, const sofia::SofiaParams& params, bool force = false)
			-> std::future<std::string>;
		auto changeProjectAsync(const std::string& projectUUID, const std::string &name)
			-> std::future<void>;

		auto getProjectAsync(const std::string& projectUUID) -> Project;
		auto getRequests(const std::string& projectUUID) -> std::vector<Request>;
		auto getRequest(const std::string& projectUUID, const std::string& requestUUID) -> std::vector<Request>;

		static constexpr auto heartbeatIntervalSeconds = 5.0f;

	private:
		static auto getServerStatusState(ServerConnectionDescription connectionDescription) -> ServerState;
		static auto getIsServerBusy(ServerConnectionDescription connectionDescription) -> bool;
		static auto getProjects(ServerConnectionDescription connectionDescription)
			-> std::optional<std::vector<Project>>;
		static auto downloadFile(ServerConnectionDescription connectionDescription, std::string fileUUID,
								 std::filesystem::path targetDirectoryPath) -> std::filesystem::path;
		static auto deleteProject(ServerConnectionDescription connectionDescription, const std::string& projectUUID)
			-> void;
		static auto changeProject(ServerConnectionDescription connectionDescription, const std::string& projectUUID, const std::string& projectName)
			-> void;

		static auto uploadFile(
			ServerConnectionDescription connectionDescription, const std::filesystem::path& sourceFile,
							   const std::string& projectName,
							   UploadFeedback& uploadFeedback) -> UploadResult;

		auto startSearch(ServerConnectionDescription connectionDescription, const std::string& projectUUID,
						 const sofia::SofiaParams& params, bool force = false) -> std::string;
		// IP or Hostname
		ServerConnectionDescription serverConnectionDescription_{ .port = "5051",
																  .ipHost = "localhost",
																  .name = "localhost" };

		ServerState lastServerState_{ ServerHealthState::unknown, ServerBusyState::unknown };

		bool lastHeartbeatDone_ = true;
		float secondsSinceLastHeartbeat_ = heartbeatIntervalSeconds;

		std::future<ServerState> heartbeatFuture_;
	};
} // namespace b3d::tools::project
