#include "SoFiaSearch.h"

#include "GizmoHelper.h"
#include "framework/ApplicationContext.h"
#include "framework/ModalViewBase.h"

#include "SoFiaSearchView.h"

SoFiaSearch::SoFiaSearch(ApplicationContext& applicationContext)
	: UpdatableComponentBase(applicationContext), RendererExtensionBase(applicationContext)
{
	sofiaSearchView_ = std::make_unique<SoFiaSearchView>(applicationContext, applicationContext.getMainDockspace());

	applicationContext.addMenuToggleAction(
		showSearchWindow_, [&](const bool isOn) { isOn ? sofiaSearchView_->open() : sofiaSearchView_->close(); },
		"Tools",
		"SoFiA");

	applicationContext.addUpdatableComponent(this);
	applicationContext.addRendererExtensionComponent(this);

	gizmoHelper_ = applicationContext.getGizmoHelper();
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

	owl::AffineSpace3f worldTransform = owl::AffineSpace3f::scale(10);
	
	// transform_ ist das transform des gizmos
	gizmoHelper_->drawBoundGizmo(transform_, worldTransform, { 1, 1, 1 });
}

auto SoFiaSearch::update() -> void
{
	if (showSearchWindow_)
	{
		sofiaSearchView_->draw();
	}
}
