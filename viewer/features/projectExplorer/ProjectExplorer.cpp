#include "ServerClient.h"
#include "ServerFileProvider.h"

#include "framework/ApplicationContext.h"

#include "ProjectExplorerController.h"

#include "ProjectExplorer.h"

using namespace b3d::tools::project;

ProjectExplorer::ProjectExplorer(ApplicationContext& applicationContext) : RendererExtensionBase(applicationContext)
{
	projectExplorerController_ = std::make_unique<ProjectExplorerController>(applicationContext, *this, projects_);
	serverFileProvider_ = std::make_unique<ServerFileProvider>("./", appContext_->serverClient);
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

	// TODO: Missing loading to renderer!
	if (loadAndShowViewPromise_ && loadAndShowViewFuture_.valid())
	{
		if (loadFileFuture_.valid())
		{
			using namespace std::chrono_literals;
			if (loadFileFuture_.wait_for(0s) == std::future_status::ready)
			{
				// Don't care about the result, just that it is done
				const auto success = loadFileFuture_.get();
				loadAndShowViewPromise_->set_value();
				loadAndShowViewPromise_.reset();

				// Shared Futures are only invalid when they are default constructed
				loadAndShowViewFuture_ = {};
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

auto ProjectExplorer::loadAndShowFile(const std::string fileUUID) -> std::shared_future<void>
{
	// To be 100% safe we need to protect loadAndShowViewFuture_. But the updates are called synchronously so it
	// should be fine.
	if (loadAndShowViewPromise_ && loadAndShowViewFuture_.valid())
	{
		return loadAndShowViewFuture_;
	}

	loadFileFuture_ = serverFileProvider_->loadFileFromServerAsync(fileUUID, false);

	loadAndShowViewPromise_ = std::make_unique<std::promise<void>>();
	loadAndShowViewFuture_ = loadAndShowViewPromise_->get_future();
	return loadAndShowViewFuture_;
}
