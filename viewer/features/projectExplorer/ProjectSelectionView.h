#pragma once

#include <future>

#include "framework/ModalViewBase.h"

namespace b3d::tools::project
{
	struct Project;
}

class ProjectSelectionView final : public ModalViewBase
{
public:
	struct Model
	{
		std::string selectedProjectUUID {};
		std::vector<b3d::tools::project::Project>* projects{};
	};


	explicit ProjectSelectionView(ApplicationContext& applicationContext,
								  std::function<std::shared_future<void>()> refreshProjectsFunction,
								  std::function<void(ModalViewBase* self)> onOpenCallback,
								  std::function<void(ModalViewBase* self)> onSubmitCallback);

	auto onDraw() -> void override;

	auto setModel(Model model) -> void;

	auto getSelectedProjectUUID() -> std::string;

private:
	auto projectsAvailable() const -> bool;
	auto requestOngoing() const -> bool;

	Model model_ {};
	std::string availableProjectString_ = "No projects available!";
	
	std::function<std::shared_future<void>()> refreshProjectsFunction_ {};
	std::shared_future<void> refreshProjectsFuture_ {};
};
