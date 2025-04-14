#include "framework/ApplicationContext.h"
#include "framework/ModalViewBase.h"

#include "ProjectExplorer.h"
#include "ProjectExplorerView.h"
#include "ProjectExplorerController.h"

#include <RenderData.h>

namespace
{
	auto toMat4(const owl::AffineSpace3f& value) -> glm::mat4
	{
		return glm::mat4{ { value.l.vx.x, value.l.vx.y, value.l.vx.z, 0.0f },
						  { value.l.vy.x, value.l.vy.y, value.l.vy.z, 0.0f },
						  { value.l.vz.x, value.l.vz.y, value.l.vz.z, 0.0f },
						  { value.p.x, value.p.y, value.p.z, 1.0f } };
	}
} // namespace


ProjectExplorerController::ProjectExplorerController(ApplicationContext& applicationContext,
													 ProjectExplorer& projectExplorer,
													 std::vector<b3d::tools::project::Project>& projects)
	: UpdatableComponentBase(applicationContext), RendererExtensionBase{ applicationContext },
	  projectExplorer_(&projectExplorer), projects_(&projects)
{
	projectExplorerView_ = std::make_unique<ProjectExplorerView>(
		applicationContext, applicationContext.getMainDockspace(), [&] { }, [&] {},
		[&](const std::string& fileUUID) { return projectExplorer_->loadAndShowFile(fileUUID); },
		[&]() { return projectExplorer_->refreshProjects(); });

	applicationContext.addMenuToggleAction(
		showExplorerWindow_, [&](const bool isOn)
		{ isOn ? projectExplorerView_->open() : projectExplorerView_->close(); }, "Tools", "Projects");
}

ProjectExplorerController::~ProjectExplorerController() = default;

auto ProjectExplorerController::setProjects(std::vector<b3d::tools::project::Project>* projects) -> void
{
	projects_ = projects;
	projectExplorerView_->setModel(ProjectExplorerView::Model{ projects_ });
}
auto ProjectExplorerController::updateRenderingData(b3d::renderer::RenderingDataWrapper& renderingData) -> void
{
	renderingData_ = &renderingData;
}

auto ProjectExplorerController::update() -> void
{
	showExplorerWindow_ = projectExplorerView_->isOpen();

	if (showExplorerWindow_)
	{
		projectExplorerView_->draw();

		if (renderingData_)
		{
			const auto hasData =
				(renderingData_->buffer.get<b3d::renderer::VolumeTransform>("volumeTransform") != nullptr) and
				(renderingData_->buffer.get<b3d::tools::renderer::nvdb::RuntimeVolumeData>("runtimeVolumeData") !=
				 nullptr);
			if (hasData)
			{


				const auto volumeTransform =
					renderingData_->buffer.get<b3d::renderer::VolumeTransform>("volumeTransform");
				const auto runtimeVolumeData =
					renderingData_->buffer.get<b3d::tools::renderer::nvdb::RuntimeVolumeData>("runtimeVolumeData");
				const auto trs = volumeTransform->worldMatTRS * runtimeVolumeData->volume.renormalizeScale;

				projectExplorerView_->setVolumeTransform(trs);
			}
		}
	}

	const auto isConnectedToAnyServer = applicationContext_->serverClient_.getLastServerStatusState().health ==
		b3d::tools::project::ServerHealthState::ok;

	if (not isConnectedToAnyServer)
	{
		// setProjects(nullptr);
	}
}
