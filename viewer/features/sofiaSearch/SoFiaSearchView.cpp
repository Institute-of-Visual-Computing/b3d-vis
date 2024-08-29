#include "SoFiaSearchView.h"

#include "GizmoHelper.h"
#include "framework/ApplicationContext.h"


SoFiaSearchView::SoFiaSearchView(ApplicationContext& appContext, Dockspace* dockspace)
	: DockableWindowViewBase(appContext, "SoFiA-Search", dockspace, WindowFlagBits::none)
{

	gizmoHelper_ = appContext.getGizmoHelper();

	model_.params.setOrReplace("reliability.minSNR", "3.0");
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
	owl::AffineSpace3f worldTransform = owl::AffineSpace3f::scale(10);

	// transform_ ist das transform des gizmos
	gizmoHelper_->drawBoundGizmo(transform_, worldTransform, { 1, 1, 1 });

	// ImGui::BeginChild("sofia-params");
	// ImGui::InputFloat("reliability.minSNR", &minSNR_);
	ImGui::SliderFloat("reliability.minSNR", &minSNR_, 0.0f, 10.0f);


	ImGui::BeginDisabled(true);

	auto scale = owl::vec3f{ length(transform_.l.vx), length(transform_.l.vx), length(transform_.l.vx) };
	auto position = transform_.p;

	auto lower = owl::vec3f{ -.5f, -.5f, -.5f };
	auto upper = owl::vec3f{ .5f, .5f, .5f };

	owl::vec3f lowerPos = xfmPoint(transform_, lower) + owl::vec3f{ .5, .5, .5 };
	owl::vec3f upperPos = xfmPoint(transform_, upper) + owl::vec3f{ .5, .5, .5 };

	owl::box3f regionBox = intersection(owl::box3f{ lowerPos, upperPos }, owl::box3f{ { 0, 0, 0 }, { 1, 1, 1 } });


	ImGui::InputFloat3("Scale", &scale.x);
	ImGui::InputFloat3("Position", &position.x);

	ImGui::InputFloat3("LowerTransformed", &regionBox.lower.x);
		ImGui::InputFloat3("UpperTransformed", &regionBox.upper.x);

	ImGui::EndDisabled();
	//ImGui::EndChild();
}
