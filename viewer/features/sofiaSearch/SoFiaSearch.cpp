#include "SoFiaSearch.h"

#include "GizmoHelper.h"
#include "RenderData.h"
#include "framework/ApplicationContext.h"

#include "ParamsSerializer.h"
#include "SoFiaSearchView.h"

SoFiaSearch::SoFiaSearch(ApplicationContext& applicationContext)
	: UpdatableComponentBase(applicationContext), RendererExtensionBase(applicationContext)
{
	sofiaSearchView_ = std::make_unique<SoFiaSearchView>(applicationContext, applicationContext.getMainDockspace(),
														 [&]()
														 {
															 b3d::tools::sofia::SofiaParams params;
															 serialize(sofiaSearchView_->getModel().params, params);
															 startSearch(params);
														 });

	applicationContext.addMenuToggleAction(
		showSearchWindow_, [&](const bool isOn) { isOn ? sofiaSearchView_->open() : sofiaSearchView_->close(); },
		"Tools", "SoFiA");

	applicationContext.addUpdatableComponent(this);
	applicationContext.addRendererExtensionComponent(this);
}

SoFiaSearch::~SoFiaSearch() = default;

auto SoFiaSearch::initializeResources() -> void
{
}

auto SoFiaSearch::deinitializeResources() -> void
{
}

auto SoFiaSearch::updateRenderingData(b3d::renderer::RenderingDataWrapper& renderingData) -> void
{
	const auto& volumeTransform = renderingData.data.volumeTransform;
	const auto& runtimeVolumeData = renderingData.data.runtimeVolumeData;

	auto& model = sofiaSearchView_->getModel();


	if (applicationContext_->selectedProject_.has_value() &&
		runtimeVolumeData.volume.state == b3d::tools::renderer::nvdb::RuntimeVolumeState::ready)
	{

		model.interactionEnabled = true;

		const auto trs = volumeTransform.worldMatTRS * runtimeVolumeData.volume.renormalizeScale *
			owl::AffineSpace3f::scale(runtimeVolumeData.originalIndexBox.size());
		model.worldTransform = trs;
	}
	else
	{
		model.interactionEnabled = false;
	}
}

auto SoFiaSearch::startSearch(b3d::tools::sofia::SofiaParams params) -> void
{
	appContext_->serverClient_.startSearchAsync(appContext_->selectedProject_.value().projectUUID, params, false);
}

auto SoFiaSearch::update() -> void
{
	if (showSearchWindow_)
	{
		sofiaSearchView_->draw();
	}
}
