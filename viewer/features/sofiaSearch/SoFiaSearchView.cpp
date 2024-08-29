#include "SoFiaSearchView.h"

#include "framework/ApplicationContext.h"

SoFiaSearchView::SoFiaSearchView(ApplicationContext& appContext, Dockspace* dockspace)
	: DockableWindowViewBase(appContext, "SoFiA-Search", dockspace, WindowFlagBits::none)
{
	
}

SoFiaSearchView::~SoFiaSearchView() = default;

auto SoFiaSearchView::setModel(Model model) -> void
{
	model_ = std::move(model);
}

auto SoFiaSearchView::getModel() const -> const Model&
{
	return model_;
}

auto SoFiaSearchView::onDraw() -> void
{
	ImGui::Text("SoFiA-Search");
}
