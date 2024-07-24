#pragma once

#include <filesystem>
#include <future>

#include "RenderFeature.h"

#include <vector>

#include "Client.h"
#include "imgui.h"

namespace b3d::tools::projectexplorer
{
	struct Project;
}

namespace b3d::renderer
{
	
	class NanoRenderer;

	class ServerClientFeature final : public RenderFeature
	{
	public:
		explicit ServerClientFeature(const std::string& name, b3d::renderer::NanoRenderer* renderer);

		auto onInitialize() -> void override;

		auto gui() -> void override;

		struct ParamsData
		{
			// TODO: put your computed params here
		};

		[[nodiscard]] auto getParamsData() -> ParamsData;

		[[nodiscard]] auto hasGui() const -> bool override;
	private:

		auto requestOngoing() -> bool;


		b3d::tools::server_client::Client serverClient;


		std::vector<b3d::tools::projectexplorer::Project> projects;


		std::unique_ptr<std::future<std::filesystem::path>> fileRequest;
		std::unique_ptr<std::future<void>> loadingRequest;



		std::unique_ptr<std::future<std::vector<b3d::tools::projectexplorer::Project>>> projectLoadingRequest;
		
		b3d::renderer::NanoRenderer* renderer_;

		std::string selectedProjectUUID_;

		auto selectProject(const tools::projectexplorer::Project& project) -> void;

		auto loadNvdb(const std::string &nvdbFileGuid) -> void;
	};

} // namespace b3d::renderer
