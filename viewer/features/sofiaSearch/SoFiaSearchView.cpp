#include "SoFiaSearchView.h"

#include "GizmoHelper.h"
#include "framework/ApplicationContext.h"
#include "IconsLucide.h"

namespace
{
	const auto lower = owl::vec3f{ -.5f, -.5f, -.5f };
	const auto upper = owl::vec3f{ .5f, .5f, .5f };
	const auto unityBoxSize = owl::vec3f{ 1.0f };
	const auto unitBox = owl::box3f{0.0f, unityBoxSize };

	auto HelpMarker(const char* desc) -> void
	{
		ImGui::TextDisabled("(?)");
		if (ImGui::BeginItemTooltip())
		{
			ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
			ImGui::TextUnformatted(desc);
			ImGui::PopTextWrapPos();
			ImGui::EndTooltip();
		}
	}

	template <typename T>
	auto ResetButtonOnSameLine(float buttonSizeX, T* valuePointer, T defaultValue) -> bool
	{
		auto changed = false;
		ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - buttonSizeX);
		if (ImGui::Button(ICON_LC_UNDO_2))
		{
			changed = true;
			*valuePointer = defaultValue;
		}
		return changed;
	}


	auto DragIntInputWidget(const char* label, int* value, int v_speed, int v_min, int v_max, int defaultValue,
						 float btnSizeX, const char* helpText = nullptr,
						 bool oddOnly = false)
	-> bool
	{
		auto changed = false;
		ImGui::PushID(label);
		if (ImGui::DragInt(label, value, v_speed, v_min, v_max, "%d", ImGuiSliderFlags_AlwaysClamp) && oddOnly)
		{
			changed = true;
			if (*value > v_min + 1 && *value % 2 == 0)
			{
				*value -= 1;
			}
		}
		if (helpText != nullptr)
		{
			ImGui::SameLine();
			HelpMarker(helpText);
		}
		changed = ResetButtonOnSameLine(btnSizeX, value, defaultValue);
		
		ImGui::PopID();
		return changed;
	}

	auto DragFloatInputWidget(const char* label, float* value, float v_speed, float v_min, float v_max,
							  float defaultValue, float btnSizeX, const char* helpText = nullptr)-> bool
	{
		auto changed = false;
		ImGui::PushID(label);
		changed = ImGui::DragFloat("Threshold", value, v_speed, v_min, v_max, "%.2f",
						 ImGuiSliderFlags_None);
		if (helpText != nullptr)
		{
			ImGui::SameLine();
			HelpMarker(helpText);
		}
		changed = ResetButtonOnSameLine(btnSizeX, value, defaultValue);
		ImGui::PopID();
		return changed;
	}
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

	const auto btnDefaultSize = ImGui::CalcTextSize(ICON_LC_UNDO_2).x + ImGui::GetStyle().FramePadding.x * 2.0f;
	
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
		ImGui::Checkbox("Enabled##contsub", &model_.params.contsub.enable);
		ImGui::SameLine();
		HelpMarker("If enabled, SoFiA will try to subtract any residual continuum emission from the data cube prior to source finding by fitting and subtracting a polynomial of order 0 (offset) or 1 (offset + slope). The order of the polynomial is defined by contsub.order.");
		if (!model_.params.contsub.enable)
		{
			ImGui::BeginDisabled(true);
		}

		ImGui::Text("Order");
		ImGui::SameLine();
		HelpMarker("Order of the polynomial used for the continuum subtraction. 0 = constant, 1 = linear.");
		ImGui::RadioButton("0", &model_.params.contsub.order, 0);
		ImGui::SameLine();
		ImGui::RadioButton("1", &model_.params.contsub.order, 1);
		ResetButtonOnSameLine(btnDefaultSize, &model_.params.contsub.order, 0);

		DragIntInputWidget("Padding", &model_.params.contsub.padding, 1, 0, 1000 * 1000, 3, btnDefaultSize, "The amount of additional padding (in channels) applied to either side of channels excluded from the fit.");
		DragIntInputWidget("Shift", &model_.params.contsub.shift, 1, 1, 1000 * 1000, 4, btnDefaultSize, "The number of channels by which the spectrum will be shifted (symmetrically in both directions) before self-subtraction.");

		DragFloatInputWidget("Threshold", &model_.params.contsub.threshold, 0.1f, 0.0f, 1000.0f, 2.0f, btnDefaultSize,
							 "Relative clipping threshold. All channels with a flux density > contsub.threshold times the noise will be clipped and excluded from the polynomial fit.");

		if (!model_.params.contsub.enable)
		{
			ImGui::EndDisabled();
		}
	}

	ImGui::BeginDisabled(true);
	if (ImGui::CollapsingHeader("Preconditioning Flagging", ImGuiTreeNodeFlags_None))
	{
		const char* items[] = { "false", "true", "channels", "pixels" };
		static auto item_current = 0;
		ImGui::Combo("Auto", &item_current, items, 4);
		ImGui::SameLine();
		HelpMarker("If set to true, SoFiA will attempt to automatically flag spectral channels and spatial pixels "
				   "affected by interference or artefacts based on their RMS noise level. If set to channels, only "
				   "spectral channels will be flagged. If set to pixels, only spatial pixels will be flagged. If set "
				   "to false, auto-flagging will be disabled. Please see the user manual for details.");
		ResetButtonOnSameLine(btnDefaultSize, &item_current, 0);
		ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - btnDefaultSize);
	
		model_.params.flag.autoMode = items[item_current];

		if (item_current == 0)
		{
			ImGui::BeginDisabled(true);
		}

		if (item_current == 0)
		{
			ImGui::EndDisabled();
		}
	}
	ImGui::EndDisabled();
	

	if (ImGui::CollapsingHeader("Preconditioning Ripple Filter", ImGuiTreeNodeFlags_None))
	{
		ImGui::Checkbox("Enabled##Ripple", &model_.params.ripple.enable);
		ImGui::SameLine();
		HelpMarker("If set to true, then the ripple filter will be applied to the data cube prior to source finding. The filter works by measuring and subtracting either the mean or median across a running window. This can be useful if a DC offset or spatial/spectral ripple is present in the data.");
		if (!model_.params.ripple.enable)
		{
			ImGui::BeginDisabled(true);
		}

		DragIntInputWidget("Grid XY", &model_.params.ripple.gridXY, 2, 0, 1001, 0, btnDefaultSize,
										  "Spatial grid separation in pixels for the running window used in the ripple filter. The value must be an odd integer value and specifies the spatial step by which the window is moved. Alternatively, it can be set to 0, in which case it will default to half the spatial window size (see rippleFilter.windowXY).", true);

		DragIntInputWidget("Grid Z", &model_.params.ripple.gridZ, 2, 0, 1001, 0, btnDefaultSize, "Spectral grid separation in channels for the running window used in the ripple filter. The value must be an odd integer value and specifies the spectral step by which the window is moved. Alternatively, it can be set to 0, in which case it will default to half the spectral window size (see rippleFilter.windowZ).", true);
		

		ImGui::Checkbox("Interpolate", &model_.params.ripple.interpolate);
		ImGui::SameLine();
		HelpMarker("Controls whether the mean or median should be measured and subtracted in the running window of the "
				   "ripple filter. The median is strongly recommended, as it is more robust.");
		if(ResetButtonOnSameLine(btnDefaultSize, &model_.params.ripple.interpolate, false))
		{
			model_.params.ripple.interpolate = false;
		}

		const char* rippleStatisticItems[] = { "median", "mean" };
		static auto rippleStatisticItem = 0;
		ImGui::Text("Statistic");
		ImGui::SameLine();
		HelpMarker("Controls whether the mean or median should be measured and subtracted in the running window of the ripple filter. The median is strongly recommended, as it is more robust.");
		auto statisticChanged = ImGui::RadioButton("Median", &rippleStatisticItem, 0);
		ImGui::SameLine();
		statisticChanged = statisticChanged || ImGui::RadioButton("Mean", &rippleStatisticItem, 1);
		ImGui::PushID("rippleStatistics");
		statisticChanged = statisticChanged || ResetButtonOnSameLine(btnDefaultSize, &rippleStatisticItem, 0);
		ImGui::PopID();
		if (statisticChanged)
		{
			model_.params.ripple.statistic = rippleStatisticItems[rippleStatisticItem];
		}

		DragIntInputWidget(
			"Window XY", &model_.params.ripple.windowXY, 2, 1, 100001, 1, btnDefaultSize,
						"Spatial size in pixels of the running window used in the ripple filter. The size must be an "
						"odd integer number.",
						true);
		DragIntInputWidget(
			"Window Z", &model_.params.ripple.windowZ, 2, 1, 100001, 1, btnDefaultSize,
			"Spectral size in channels of the running window used in the ripple filter. The size must be an odd integer number.",
			true);

		if (!model_.params.ripple.enable)
		{
			ImGui::EndDisabled();
		}
	}

	if (ImGui::CollapsingHeader("Preconditioning Noise Scaling", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Checkbox("Enabled##Noise", &model_.params.scaleNoise.enable);

		if (!model_.params.scaleNoise.enable)
		{
			ImGui::BeginDisabled(true);
		}

		const char* fluxRangeItems[] = { "negative", "positive", "full" };
		static auto fluxRangeItem = 0;




		if (!model_.params.scaleNoise.enable)
		{
			ImGui::EndDisabled();
		}
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
