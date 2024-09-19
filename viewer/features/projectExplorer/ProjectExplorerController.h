#pragma once

#include <future>
#include <memory>
#include <string>
#include <vector>

#include "framework/RendererExtensionBase.h"
#include "framework/UpdatableComponentBase.h"

class OpenFileDialogView;

namespace b3d::tools::project
{
	struct Project;
}

class ProjectExplorer;
class ProjectExplorerView;
class ProjectSelectionView;

class ProjectExplorerController final : public UpdatableComponentBase, public RendererExtensionBase
{
public:
	ProjectExplorerController(ApplicationContext& applicationContext, ProjectExplorer& projectExplorer,
							  std::vector<b3d::tools::project::Project>& projects);
	~ProjectExplorerController() override;

	auto update() -> void override;

	auto setProjects(std::vector<b3d::tools::project::Project>* projects) -> void;
	auto initializeResources() -> void override{};
	auto deinitializeResources() -> void override{};
	auto updateRenderingData(b3d::renderer::RenderingDataWrapper& renderingData) -> void override;

private:
	b3d::renderer::RenderingDataWrapper* renderingData_{nullptr};

	ProjectExplorer* projectExplorer_;
	std::unique_ptr<ProjectExplorerView> projectExplorerView_;
	std::unique_ptr<ProjectSelectionView> projectSelectionView_;
	std::unique_ptr<OpenFileDialogView> openFileDialogView_;

	bool showExplorerWindow_{ true };

	std::vector<b3d::tools::project::Project>* projects_;
	std::string selectedProjectUUID_{};
};
