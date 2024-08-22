#pragma once

#include <future>
#include <memory>
#include <vector>

#include "framework/RendererExtensionBase.h"

namespace b3d::tools::project
{
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

private:
	auto initializeResources() -> void override;
	auto deinitializeResources() -> void override;
	auto updateRenderingData(b3d::renderer::RenderingDataWrapper& renderingData) -> void override;

	std::unique_ptr<ProjectExplorerController> projectExplorerController_;
	std::vector<b3d::tools::project::Project> projects_;

	std::future<std::optional<std::vector<b3d::tools::project::Project>>> projectsRequestFuture_;

	std::unique_ptr<std::promise<void>> projectsViewPromise_;
	std::shared_future<void> projectsViewSharedFuture_;
};
