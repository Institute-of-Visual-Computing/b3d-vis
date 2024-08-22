#pragma once

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

	ProjectExplorerView(ApplicationContext& appContext, Dockspace* dockspace, std::function<void()> showSelectionModal);
	~ProjectExplorerView() override;

	auto setModel(Model model) -> void;

private:
	auto onDraw() -> void override;
	auto projectAvailable() const -> bool;

	Model model_;
	std::function<void()> showSelectionModal_;
};
