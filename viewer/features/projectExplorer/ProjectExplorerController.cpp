#include "framework/ApplicationContext.h"
#include "framework/ModalViewBase.h"

#include "ProjectExplorer.h"
#include "ProjectExplorerView.h"
#include "ProjectSelectionView.h"

#include "ProjectExplorerController.h"

#include "../fileDialog/OpenFileDialogView.h"

ProjectExplorerController::ProjectExplorerController(ApplicationContext& applicationContext,
                                                     ProjectExplorer& projectExplorer,
                                                     std::vector<b3d::tools::project::Project>& projects)
	: UpdatableComponentBase(applicationContext), projectExplorer_(&projectExplorer), projects_(&projects)
{
	projectExplorerView_ = std::make_unique<ProjectExplorerView>(
		applicationContext, applicationContext.getMainDockspace(), [&] { projectSelectionView_->open(); },
		[&] { openFileDialogView_->open(); },
		[&](const std::string& fileUUID) { return projectExplorer_->loadAndShowFile(fileUUID); },
		[&]() { return projectExplorer_->refreshProjects();
		}
		);

	projectSelectionView_ = std::make_unique<ProjectSelectionView>(
		applicationContext, [&]() { return projectExplorer_->refreshProjects();
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

					applicationContext_->selectedProject_ = *it;

					projectExplorer_->setCurrentProject(selectedProjectUUID);
					//projectExplorerView_->setModel({ &*it });
				}
			}
		});

	openFileDialogView_ = std::make_unique<OpenFileDialogView>(
		applicationContext,
		[&](ModalViewBase* self)
		{
			reinterpret_cast<OpenFileDialogView*>(self)->setViewParams(std::filesystem::current_path(), {".nvdb"} );
		},
			[&](ModalViewBase* self)
		{
			const auto selectedPath = reinterpret_cast<OpenFileDialogView*>(self)->getModel().selectedPath_;
			if (!selectedPath.empty())
			{
				projectExplorer_->loadAndShowFileWithPath(selectedPath);
			}
		}
	);

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
	projectExplorerView_->setModel(ProjectExplorerView::Model{projects_});
}

auto ProjectExplorerController::update() -> void
{
	if (showExplorerWindow_)
	{
		projectExplorerView_->draw();
	}

	const auto isConnectedToAnyServer =
		applicationContext_->serverClient_.getLastServerStatusState() == b3d::tools::project::ServerStatusState::ok;

	if(not isConnectedToAnyServer)
	{
		setProjects(nullptr);
	}
	projectSelectionView_->draw();
	openFileDialogView_->draw();
}
