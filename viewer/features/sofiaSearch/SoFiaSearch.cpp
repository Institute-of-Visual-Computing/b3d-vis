#include "SoFiaSearch.h"

#include "framework/ApplicationContext.h"
#include "framework/ModalViewBase.h"

#include "SoFiaSearchView.h"

SoFiaSearch::SoFiaSearch(ApplicationContext& applicationContext) : UpdatableComponentBase(applicationContext)
{
	sofiaSearchView_ = std::make_unique<SoFiaSearchView>(applicationContext, applicationContext.getMainDockspace());

	applicationContext.addMenuToggleAction(
		showSearchWindow_, [&](const bool isOn) { isOn ? sofiaSearchView_->open() : sofiaSearchView_->close(); },
		"Tools",
		"SoFiA");
	applicationContext.addUpdatableComponent(this);
}

SoFiaSearch::~SoFiaSearch() = default;

auto SoFiaSearch::update() -> void
{
	if (showSearchWindow_)
	{
		sofiaSearchView_->draw();
	}
}
