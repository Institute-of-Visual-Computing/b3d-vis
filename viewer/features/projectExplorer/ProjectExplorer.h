#pragma once

#include <future>
#include <memory>
#include <vector>

#include "framework/RendererExtensionBase.h"
#include "RuntimeDataSet.h"

namespace b3d::tools::project
{
	class ServerFileProvider;
	struct Project;
}

class ProjectSelectionView;
class ProjectExplorerController;
class RenderingDataWrapper;

class ProjectExplorer final : public RendererExtensionBase
{
public:
	explicit ProjectExplorer(ApplicationContext& applicationContext);
	~ProjectExplorer() override;
	
	auto refreshProjects() -> std::shared_future<void>;
	auto loadAndShowFile(const std::string fileUUID) -> std::shared_future<void>;
	auto loadAndShowFileWithPath(std::filesystem::path absoluteFilePath) -> std::shared_future<void>;

private:
	friend ProjectExplorerController;
	auto initializeResources() -> void override;
	auto deinitializeResources() -> void override;
	auto updateRenderingData(b3d::renderer::RenderingDataWrapper& renderingData) -> void override;
	auto setCurrentProject(const std::string& projectUUID) -> void;

	std::string requestedVolumeUUid;

	std::unique_ptr<ProjectExplorerController> projectExplorerController_;
	std::unique_ptr<b3d::tools::project::ServerFileProvider> serverFileProvider_;
	std::vector<b3d::tools::project::Project> projects_;

	std::future<std::optional<std::vector<b3d::tools::project::Project>>> projectsRequestFuture_;
	std::unique_ptr<std::promise<void>> projectsViewPromise_;
	std::shared_future<void> projectsViewSharedFuture_;

	std::future<bool> loadFileFuture_;
	std::unique_ptr<std::promise<void>> loadAndShowViewPromise_;
	std::shared_future<void> loadAndShowViewFuture_;

	bool projectChanged_{ false };

	
	cudaStream_t stream_{ 0 };
};
