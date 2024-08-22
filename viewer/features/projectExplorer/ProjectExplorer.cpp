#include "ServerClient.h"

#include "framework/ApplicationContext.h"

#include "ProjectExplorerController.h"

#include "ProjectExplorer.h"

using namespace b3d::tools::project;

ProjectExplorer::ProjectExplorer(ApplicationContext& applicationContext) : RendererExtensionBase(applicationContext)
{
	projectExplorerController_ = std::make_unique<ProjectExplorerController>(applicationContext, *this, projects_);
	applicationContext.addUpdatableComponent(projectExplorerController_.get());
	applicationContext.addRendererExtensionComponent(this);
}

ProjectExplorer::~ProjectExplorer() = default;


auto ProjectExplorer::initializeResources() -> void
{
	
}

auto ProjectExplorer::deinitializeResources() -> void
{

}

auto ProjectExplorer::updateRenderingData(b3d::renderer::RenderingDataWrapper& renderingData) -> void
{
	if (projectsViewPromise_ && projectsViewSharedFuture_.valid())
	{
		if (projectsRequestFuture_.valid())
		{
			using namespace std::chrono_literals;
			if (projectsRequestFuture_.wait_for(0s) == std::future_status::ready)
			{
				const auto projects = projectsRequestFuture_.get();
				if (projects.has_value() && !projects->empty())
				{
					projects_ = *projects;
					projectExplorerController_->setProjects(&projects_);
				}
				projectsViewPromise_->set_value();
				projectsViewPromise_.reset();

				// Shared Futures are only invalid when they are default constructed
				projectsViewSharedFuture_ = {};
			}
		}
	}
}

auto ProjectExplorer::refreshProjects() -> std::shared_future<void>
{
	// To be 100% safe we need to protect projectsViewSharedFuture_. But the updates are called synchronously so it should be fine.
	if (projectsViewPromise_ && projectsViewSharedFuture_.valid())
	{
		return projectsViewSharedFuture_;
	}

	projectsRequestFuture_ = appContext_->serverClient.getProjectsAsync();

	projectsViewPromise_ = std::make_unique<std::promise<void>>();
	projectsViewSharedFuture_ = projectsViewPromise_->get_future();
	return projectsViewSharedFuture_;
}
