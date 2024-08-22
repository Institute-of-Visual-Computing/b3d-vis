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
		b3d::tools::project::Project* project {};
	};

	ProjectExplorerView(ApplicationContext& appContext, Dockspace* dockspace, std::function<void()> showSelectionModal,
						std::function<std::shared_future<void>(const std::string& fileUUID)> loadAndShowFunction);
	~ProjectExplorerView() override;

	auto setModel(Model model) -> void;

private:
	auto onDraw() -> void override;
	auto projectAvailable() const -> bool;

	Model model_;
	std::function<void()> showSelectionModal_;

	std::function<std::shared_future<void>(const std::string& fileUUID)> loadAndShowFunction_{};
	std::shared_future<void> loadAndShowFileFuture_{};
};
