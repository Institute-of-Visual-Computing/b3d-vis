#include "SoFiaSearchView.h"

#include "framework/ApplicationContext.h"


SoFiaSearchView::SoFiaSearchView(ApplicationContext& appContext, Dockspace* dockspace,
								 std::function<void()> startSearchFunction)
	: DockableWindowViewBase(appContext, "SoFiA-Search", dockspace, WindowFlagBits::none),
	  startSearchFunction_(std::move(startSearchFunction))
{
	
}

SoFiaSearchView::~SoFiaSearchView() = default;

auto SoFiaSearchView::setModel(Model model) -> void
{
	model_ = std::move(model);
}

auto SoFiaSearchView::getModel() -> Model&
{
	return model_;
}

auto SoFiaSearchView::onDraw() -> void
{
	const auto disableInteraction = !applicationContext_->selectedProject_.has_value();
	const auto hasProject = applicationContext_->selectedProject_.has_value();

	if (disableInteraction)
	{
		ImGui::Text("No project selected.");
		ImGui::BeginDisabled(true);
	}

	ImGui::Checkbox("Gizmo visible", &model_.showRoiGizmo);
	ImGui::SameLine();
	if (ImGui::Button("Reset Selection"))
	{
		resetSelection();
	}

	if (disableInteraction)
	{
		ImGui::EndDisabled();
	}

	const auto lower = owl::vec3f{ -.5f, -.5f, -.5f };
	const auto upper = owl::vec3f{ .5f, .5f, .5f };
	const auto lowerPos = xfmPoint(model_.transform_, lower) + owl::vec3f{ .5, .5, .5 };
	const auto upperPos = xfmPoint(model_.transform_, upper) + owl::vec3f{ .5, .5, .5 };

	model_.selectedLocalRegion =
		intersection(owl::box3f{ lowerPos, upperPos }, owl::box3f{ { 0, 0, 0 }, { 1, 1, 1 } });

	//ImGui::InputFloat3("Position", &model_.transform_.p.x);
	//ImGui::InputFloat3("LowerTransformed", &model_.selectedLocalRegion.lower.x);
	//ImGui::InputFloat3("UpperTransformed", &model_.selectedLocalRegion.upper.x);

	auto dimensions = owl::vec3i{ 0 };

	if (hasProject)
	{
		const auto& dims = applicationContext_->selectedProject_.value().fitsOriginProperties.axisDimensions;
		dimensions = { dims[0], dims[1], dims[2] };
	}
	
	model_.params.input.region.lower =
		owl::vec3i{ static_cast<int>(model_.selectedLocalRegion.lower.x * dimensions[0]),
					static_cast<int>(model_.selectedLocalRegion.lower.y * dimensions[1]),
					static_cast<int>(model_.selectedLocalRegion.lower.z * dimensions[2]) };

	model_.params.input.region.upper =
		owl::vec3i{ static_cast<int>(model_.selectedLocalRegion.upper.x * dimensions[0]),
					static_cast<int>(model_.selectedLocalRegion.upper.y * dimensions[1]),
					static_cast<int>(model_.selectedLocalRegion.upper.z * dimensions[2]) };
	

	/* Not used
	if (ImGui::CollapsingHeader("Pipeline", ImGuiTreeNodeFlags_None))
	{
		
	}
	*/

	if (ImGui::CollapsingHeader("Input", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Text("Region");

		ImGui::BeginDisabled(true);

		ImGui::DragInt3("Min", &model_.params.input.region.lower.x);
		model_.params.input.region.lower =
			owl::clamp(model_.params.input.region.lower, model_.params.input.region.upper);

		ImGui::DragInt3("Max", &model_.params.input.region.upper.x);
		model_.params.input.region.upper =
			owl::clamp(model_.params.input.region.upper, model_.params.input.region.lower, dimensions);

		ImGui::EndDisabled();
	}

	if (ImGui::CollapsingHeader("Preconditioning Continuum Substraction", ImGuiTreeNodeFlags_None))
	{
	}

	if (ImGui::CollapsingHeader("Preconditioning Flagging", ImGuiTreeNodeFlags_None))
	{
	}

	if (ImGui::CollapsingHeader("Preconditioning Ripple Filter", ImGuiTreeNodeFlags_None))
	{
	}

	if (ImGui::CollapsingHeader("Preconditioning Noise Scaling", ImGuiTreeNodeFlags_None))
	{
	}

	if (ImGui::CollapsingHeader("Source Finding", ImGuiTreeNodeFlags_None))
	{
	}

	if (ImGui::CollapsingHeader("Linking", ImGuiTreeNodeFlags_None))
	{
	}

	if (ImGui::CollapsingHeader("Reliability", ImGuiTreeNodeFlags_None))
	{
	}

	if (ImGui::CollapsingHeader("Mask Dilation", ImGuiTreeNodeFlags_None))
	{
	}

	if (ImGui::CollapsingHeader("Parametrisation", ImGuiTreeNodeFlags_None))
	{
	}

	/* Not used
	if (ImGui::CollapsingHeader("Output", ImGuiTreeNodeFlags_None))
	{
	}
	*/

	if (ImGui::Button("Search"))
	{
		startSearchFunction_();
		resetParams();
		resetSelection();
		resetSelection();
		// reset params
	}
}

auto SoFiaSearchView::resetSelection() -> void
{
	model_.transform_ = owl::AffineSpace3f{};
	model_.selectedLocalRegion = owl::box3f{};
}

auto SoFiaSearchView::resetParams() -> void
{
	model_.transform_ = owl::AffineSpace3f{};
	model_.selectedLocalRegion = owl::box3f{};
}
