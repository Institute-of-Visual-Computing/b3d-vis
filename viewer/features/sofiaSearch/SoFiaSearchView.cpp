#include "SoFiaSearchView.h"

#include "GizmoHelper.h"
#include "framework/ApplicationContext.h"

namespace
{
	const auto lower = owl::vec3f{ -.5f, -.5f, -.5f };
	const auto upper = owl::vec3f{ .5f, .5f, .5f };
	const auto unityBoxSize = owl::vec3f{ 1.0f };
	const auto unitBox = owl::box3f{0.0f, unityBoxSize };
}

auto SoFiaSearchView::SofiaParamsTyped::buildSoFiaParams() const -> b3d::tools::sofia::SofiaParams
{
	b3d::tools::sofia::SofiaParams sofiaParams;
	sofiaParams.setOrReplace("input.region",
							 std::format("{},{},{},{},{},{}", input.region.lower.x, input.region.upper.x,
										 input.region.lower.y, input.region.upper.y, input.region.lower.z,
										 input.region.upper.z));
	return sofiaParams;
}

SoFiaSearchView::SoFiaSearchView(ApplicationContext& appContext, Dockspace* dockspace,
                                 std::function<void()> startSearchFunction)
	: DockableWindowViewBase(appContext, "SoFiA-Search", dockspace, WindowFlagBits::none),
	  startSearchFunction_(std::move(startSearchFunction))
{
	gizmoHelper_ = applicationContext_->getGizmoHelper();
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
	const auto disableInteraction = !model_.interactionEnabled;
	const auto hasProject = applicationContext_->selectedProject_.has_value();

	if (disableInteraction)
	{
		ImGui::Text("Dataset not loaded.");
		ImGui::BeginDisabled(true);
	}

	ImGui::Checkbox("Show Subregion Tool", &model_.showRoiGizmo);
	ImGui::SameLine();
	if (ImGui::Button("Reset Selection"))
	{
		resetSelection();
	}

	if (disableInteraction)
	{
		ImGui::EndDisabled();
	}

	if (!disableInteraction && model_.showRoiGizmo)
	{
		gizmoHelper_->drawBoundGizmo(model_.transform, model_.worldTransform, unityBoxSize);
	}


	const auto lowerPos = xfmPoint(model_.transform, lower) + upper;
	const auto upperPos = xfmPoint(model_.transform, upper) + upper;
	model_.selectedLocalRegion = intersection(owl::box3f{ lowerPos, upperPos }, unitBox);

	model_.transform.p = model_.selectedLocalRegion.center() + lower;

	const auto scale = model_.selectedLocalRegion.span();
	model_.transform.l.vx.x = scale.x;
	model_.transform.l.vy.y = scale.y;
	model_.transform.l.vz.z = scale.z;

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

	model_.params.input.region.lower = owl::clamp(model_.params.input.region.lower, model_.params.input.region.upper);
	model_.params.input.region.upper =
		owl::clamp(model_.params.input.region.upper, model_.params.input.region.lower, dimensions);

	/* Not used
	if (ImGui::CollapsingHeader("Pipeline", ImGuiTreeNodeFlags_None))
	{

	}
	*/
	if (disableInteraction)
	{
		ImGui::BeginDisabled(true);
	}

	if (ImGui::CollapsingHeader("Input", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Text("Region");

		ImGui::BeginDisabled(true);

		ImGui::DragInt3("Min", &model_.params.input.region.lower.x);


		ImGui::DragInt3("Max", &model_.params.input.region.upper.x);


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
	}
	if (disableInteraction)
	{
		ImGui::EndDisabled();
	}
}

auto SoFiaSearchView::resetSelection() -> void
{
	model_.transform = owl::AffineSpace3f{};
	model_.selectedLocalRegion = owl::box3f{};
}

auto SoFiaSearchView::resetParams() -> void
{
	model_.transform = owl::AffineSpace3f{};
	model_.selectedLocalRegion = owl::box3f{};
}
