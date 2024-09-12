#pragma once

#include <future>

#include "framework/DockableWindowViewBase.h"

namespace b3d::tools::project
{
	struct Project;
}

class ProjectExplorerView final : public DockableWindowViewBase
{
public:
	struct Model
	{
		std::vector<b3d::tools::project::Project>* projects;
	};

	ProjectExplorerView(ApplicationContext& appContext, Dockspace* dockspace, std::function<void()> showSelectionModal,
						std::function<void()> showNvdbSelectionModal,
						std::function<std::shared_future<void>(const std::string& fileUUID)> loadAndShowFunction,
						std::function<std::shared_future<void>()> refreshProjectsFunction);
	~ProjectExplorerView() override;

	auto setModel(Model model) -> void;

private:
	auto onDraw() -> void override;
	auto projectAvailable() const -> bool;

	int selectedProjectItemIndex_{ -1 };

	Model model_{nullptr};
	std::function<void()> showSelectionModal_;
	std::function<void()> showNvdbSelectionModal_;

	std::function<std::shared_future<void>(const std::string& fileUUID)> loadAndShowFunction_{};
	std::function<std::shared_future<void>()> refreshProjectsFunction_ {};
	std::shared_future<void> refreshProjectsFuture_ {};
	std::shared_future<void> loadAndShowFileFuture_{};
};
