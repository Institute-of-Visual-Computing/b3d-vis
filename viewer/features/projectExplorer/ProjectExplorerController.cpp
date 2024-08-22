#include "framework/ApplicationContext.h"
#include "framework/ModalViewBase.h"

#include "ProjectExplorer.h"
#include "ProjectExplorerView.h"
#include "ProjectSelectionView.h"

#include "ProjectExplorerController.h"

ProjectExplorerController::ProjectExplorerController(ApplicationContext& applicationContext,
                                                     ProjectExplorer& projectExplorer,
                                                     std::vector<b3d::tools::project::Project>& projects)
	: UpdatableComponentBase(applicationContext), projectExplorer_(&projectExplorer), projects_(&projects)
{
	projectExplorerView_ = std::make_unique<ProjectExplorerView>(
		applicationContext, applicationContext.getMainDockspace(), [&] { projectSelectionView_->open(); });

	projectSelectionView_ = std::make_unique<ProjectSelectionView>(
		applicationContext, [&]() { return requestProjects();
		},
		[&](ModalViewBase* self) {
			reinterpret_cast<ProjectSelectionView*>(self)->setModel({selectedProjectUUID_, projects_});
		},
		[&](ModalViewBase* self)
		{
			const auto selectedProjectUUID = reinterpret_cast<ProjectSelectionView*>(self)->getSelectedProjectUUID();
			if (!selectedProjectUUID.empty())
			{
				selectedProjectUUID_ = selectedProjectUUID;
				
				const auto it = std::ranges::find_if(projects_->begin(), projects_->end(),
													 [&uuid = selectedProjectUUID](const b3d::tools::project::Project &currProject)
										 { return currProject.projectUUID == uuid;
				});
				if (it != projects_->end())
				{
					projectExplorerView_->setModel({ &*it });
				}
			}
		});

	applicationContext.addMenuToggleAction(
		showExplorerWindow_,
		[&](const bool isOn) { isOn ? projectExplorerView_->open() : projectExplorerView_->close(); },
		"Tools",
		"Projects");
}

ProjectExplorerController::~ProjectExplorerController() = default;

auto ProjectExplorerController::setProjects(std::vector<b3d::tools::project::Project>* projects) -> void
{
	projects_ = projects;
	projectSelectionView_->setModel({ projectSelectionView_->getSelectedProjectUUID(), projects_ });
}

auto ProjectExplorerController::update() -> void
{
	if (showExplorerWindow_)
	{
		projectExplorerView_->draw();
	}
	projectSelectionView_->draw();
}

auto ProjectExplorerController::requestProjects() -> std::shared_future<void>
{
	return projectExplorer_->refreshProjects();
}
