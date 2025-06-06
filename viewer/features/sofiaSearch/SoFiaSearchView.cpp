#include "SoFiaSearchView.h"

#include "GizmoHelper.h"
#include "IconsLucide.h"
#include "Style.h"
#include "framework/ApplicationContext.h"

#include <format>

namespace
{
	const auto lower = owl::vec3f{ -.5f, -.5f, -.5f };
	const auto upper = owl::vec3f{ .5f, .5f, .5f };
	const auto unityBoxSize = owl::vec3f{ 1.0f };
	const auto unitBox = owl::box3f{ 0.0f, unityBoxSize };

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

	auto ResetButtonOnSameLine(float buttonSizeX) -> bool
	{
		ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - buttonSizeX);
		return ui::Button(ICON_LC_UNDO_2);
	}

	template <typename T>
	auto ResetButtonOnSameLine(float buttonSizeX, T* valuePointer, T defaultValue) -> bool
	{
		if (ResetButtonOnSameLine(buttonSizeX))
		{
			*valuePointer = defaultValue;
			return true;
		}
		return false;
	}


	auto DragIntInputWidget(const char* label, int* value, int v_speed, int v_min, int v_max, int defaultValue,
							float btnSizeX, const char* helpText = nullptr, bool oddOnly = false) -> bool
	{
		auto changed = false;
		ImGui::PushID(label);
		if (ImGui::DragInt(label, value, static_cast<float>(v_speed), v_min, v_max, "%d",
						   ImGuiSliderFlags_AlwaysClamp) &&
			oddOnly)
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
							  float defaultValue, float btnSizeX, const char* helpText = nullptr,
							  const char* format = "%.2f") -> bool
	{
		auto changed = false;
		ImGui::PushID(label);
		changed = ImGui::DragFloat("Threshold", value, v_speed, v_min, v_max, format, ImGuiSliderFlags_None);
		if (helpText != nullptr)
		{
			ImGui::SameLine();
			HelpMarker(helpText);
		}
		changed = ResetButtonOnSameLine(btnSizeX, value, defaultValue);
		ImGui::PopID();
		return changed;
	}

	auto ComboWidget(const char* label, const char* itemValues[], int itemCount, int* selectedItemIndex,
					 std::string* target, int defaultItemIndex, float btnSizeX, const char* helperText = nullptr)
		-> bool
	{
		ImGui::PushID(label);
		auto changed = ImGui::Combo(label, selectedItemIndex, itemValues, itemCount);
		if (helperText != nullptr)
		{
			ImGui::SameLine();
			HelpMarker(helperText);
		}
		changed = changed || ResetButtonOnSameLine(btnSizeX, selectedItemIndex, defaultItemIndex);

		if (changed)
		{
			*target = itemValues[*selectedItemIndex];
		}
		ImGui::PopID();
		return changed;
	}


} // namespace

auto SoFiaSearchView::SofiaParamsTyped::buildSoFiaParams() -> b3d::tools::sofia::SofiaParams
{
	b3d::tools::sofia::SofiaParams sofiaParams;

	// serialize(*this, sofiaParams);
	sofiaParams.setOrReplace("input.region",
							 std::format("{},{},{},{},{},{}", input.region.lower.x, input.region.upper.x,
										 input.region.lower.y, input.region.upper.y, input.region.lower.z,
										 input.region.upper.z));
	return sofiaParams;
}

SoFiaSearchView::SoFiaSearchView(ApplicationContext& appContext, Dockspace* dockspace,
								 std::function<void()> startSearchFunction)
	: DockableWindowViewBase(appContext, "SoFiA Search", dockspace, WindowFlagBits::none),
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

	const auto& brush = ApplicationContext::getStyleBrush();
	constexpr auto containerCornerRadius = 8.0f;
	constexpr auto contentCornerRadius = 4.0f;
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, containerCornerRadius);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, containerCornerRadius);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f);

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vector2{ 24.0f, 24.0f });
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vector2{ 24.0f, 24.0f });
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, contentCornerRadius);

	ImGui::PushStyleColor(ImGuiCol_PopupBg, brush.cardBackgroundFillColorDefaultBrush);
	ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrokeColorSecondaryBrush);

	const auto font = applicationContext_->getFontCollection().getTitleFont();
	ImGui::PushFont(font);
	ImGui::Text("SoFiA Search");
	ImGui::PopFont();

	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vector2{ 24.0f, 12.0f });
	ImGui::TextWrapped("This tool utilizes SiFiA 2 in the backend to search for sources in a HI data cubes.");
	ImGui::TextLinkOpenURL("Learn more about SoFiA 2", "https://gitlab.com/SoFiA-Admin/SoFiA-2");
	ImGui::TextLinkOpenURL("Learn more about SoFiA 2 control parameters",
						   "https://gitlab.com/SoFiA-Admin/SoFiA-2/-/wikis/SoFiA-2-Control-Parameters");
	ImGui::PopStyleVar();

	if (disableInteraction)
	{
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.systemFillColorCautionBackgroundBrush);

		ImGui::BeginChild("##project_not_selected_warning", Vector2{},
						  ImGuiChildFlags_AlwaysAutoResize | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_FrameStyle);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.systemFillColorCautionBrush);
		ImGui::Text(ICON_LC_MESSAGE_SQUARE_WARNING);
		ImGui::PopStyleColor();
		ImGui::SameLine();
		ImGui::TextWrapped("No project is currently loaded. Please select or add a new project dataset in the Project "
						   "Explorer window.");
		ImGui::EndChild();
		ImGui::PopStyleColor();
	}

	const auto& style = ImGui::GetStyle();

	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vector2{ 8.0f, 8.0f });

	ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(ICON_LC_FILTER).x -
							4.0f * style.FramePadding.x - ImGui::CalcTextSize(ICON_LC_FILTER_X).x -
							2.0f * style.ItemSpacing.x);
	ImGui::BeginDisabled(not isFilterEnabled_);
	const auto filterChanged = ui::InputText("##sofia_params_filter", "SoFiA Parameter Filter", paramsFilter_.InputBuf,
											 IM_ARRAYSIZE(paramsFilter_.InputBuf));
	if (filterChanged)
	{
		paramsFilter_.Build();
	}
	ImGui::SameLine();
	if (ui::Button(ICON_LC_FILTER_X "##reset_sofia_params_filter"))
	{
		paramsFilter_.Clear();
	}
	ImGui::SetItemTooltip("Reset Filter");
	ImGui::EndDisabled();

	ImGui::SameLine();
	if (ui::ToggleButton(isFilterEnabled_, ICON_LC_FILTER "##apply_sofia_params_filter"))
	{
		isFilterEnabled_ = !isFilterEnabled_;
	}
	ImGui::SetItemTooltip("Enable/Disable Filter");
	ImGui::PopStyleVar();

	ImGui::Spacing();

	const auto footerHeight = ImGui::CalcTextSize("Submit SoFiA Search").y + style.FramePadding.y * 2.0f + 24.0f * 2.0f;
	const auto availableSize = Vector2{ ImGui::GetContentRegionAvail() };
	if (ImGui::BeginChild("sofia_search_filter_settings", Vector2{ 0.0f, availableSize.y - footerHeight }/*,
					  ImGuiChildFlags_FrameStyle*/))
	{
		drawFilterFormContent();
	}
	ImGui::EndChild();
	const auto position = ImGui::GetCursorScreenPos();
	const auto min = position - Vector2{ style.FramePadding.x + 6, 0 };
	const auto max = position + Vector2{ style.FramePadding.x + 6, style.FramePadding.y + style.ItemSpacing.y } +
		Vector2{ availableSize.x, footerHeight };
	ImGui::GetWindowDrawList()->AddRectFilled(min, max, brush.solidBackgroundFillColorSecondaryBrush);


	ImGui::BeginDisabled(disableInteraction);
	ImGui::SetCursorPos(ImGui::GetCursorPos() + Vector2{ 0.0, 24.0f - style.FrameBorderSize });
	if (ui::AccentButton("Submit SoFiA Search", Vector2{ availableSize.x, 0.0f }))
	{
		startSearchFunction_();
		resetParams();
		resetSelection();
		resetSelection();
	}
	ImGui::EndDisabled();
	ImGui::PopStyleColor(2);
	ImGui::PopStyleVar(6);
}

auto SoFiaSearchView::drawFilterFormContent() -> void
{
	const auto hasProject = applicationContext_->selectedProject_.has_value();

	constexpr auto toggleSwitchWidth = 14.0f * 4.0f + 4.0f;
	const auto disableInteraction = !hasProject;

	ImGui::BeginDisabled(disableInteraction);

	if (!disableInteraction and model_.showRoiGizmo)
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
					static_cast<int>(dimensions[2] - model_.selectedLocalRegion.upper.z * dimensions[2]) };

	model_.params.input.region.upper =
		owl::vec3i{ static_cast<int>(model_.selectedLocalRegion.upper.x * dimensions[0]),
					static_cast<int>(model_.selectedLocalRegion.upper.y * dimensions[1]),
					static_cast<int>(dimensions[2] - model_.selectedLocalRegion.lower.z * dimensions[2]) };

	model_.params.input.region.lower = owl::clamp(model_.params.input.region.lower, model_.params.input.region.upper);
	model_.params.input.region.upper =
		owl::clamp(model_.params.input.region.upper, model_.params.input.region.lower, dimensions);

	const auto& brush = ApplicationContext::getStyleBrush();
	[[maybe_unused]] const auto& style = ImGui::GetStyle();
	const auto undoButtonDefaultSize = ImGui::CalcTextSize(ICON_LC_UNDO_2).x + ImGui::GetStyle().FramePadding.x * 2.0f;

	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f);


	if (not isFilterEnabled_ or paramsFilter_.PassFilter("region sub input min max"))
	{
		if (ImGui::BeginChild("##region_input", Vector2{ 0.0f, 0.0f },
							  ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY))
		{

			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

			ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
			const auto titleWidth = ImGui::CalcTextSize("Region Input").x;
			ImGui::TextWrapped("Region Input");
			ImGui::PopFont();
			ImGui::SameLine(0.0f, ImGui::GetContentRegionAvail().x - toggleSwitchWidth - titleWidth);
			if (ui::ToggleSwitch(model_.showRoiGizmo, "##enable_region_input_settings", "off", "on"))
			{
				model_.showRoiGizmo = !model_.showRoiGizmo;
			}

			ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);
			ImGui::TextWrapped(
				"Region input confines the finder to a sub-data cube, which can lead to a faster computation.");
			ImGui::PopStyleColor();

			ImGui::BeginDisabled(!model_.showRoiGizmo);

			ui::HeadedDragInt3("Min", "##min", &model_.params.input.region.lower.x);
			ui::HeadedDragInt3("Max", "##max", &model_.params.input.region.upper.x);


			ImGui::EndDisabled();
			ImGui::PopStyleColor();
			ImGui::PopStyleVar();
		}
		ImGui::EndChild();
	}

	if (not isFilterEnabled_ or
		paramsFilter_.PassFilter("preconditioning continuum subtraction order padding shift threshold"))
	{
		if (ImGui::BeginChild("##preconditioning_continuum_subtraction", Vector2{ 0.0f, 0.0f },
							  ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY))
		{

			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

			ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
			const auto wrapPosition = ImGui::GetContentRegionAvail().x - toggleSwitchWidth;
			ImGui::PushTextWrapPos(wrapPosition);

			const auto titleWidth =
				ImGui::CalcTextSize("Preconditioning Continuum Subtraction", 0, true, wrapPosition).x;
			ImGui::TextWrapped("Preconditioning Continuum Subtraction");
			ImGui::PopTextWrapPos();
			ImGui::PopFont();
			ImGui::SameLine(0.0f, wrapPosition - titleWidth);
			if (ui::ToggleSwitch(model_.params.contsub.enable, "##enable_preconditioning_continuum_subtraction", "off",
								 "on"))
			{
				model_.params.contsub.enable = !model_.params.contsub.enable;
			}

			ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);
			ImGui::TextWrapped(
				"If enabled, SoFiA will try to subtract any residual continuum emission from the data cube prior to "
				"source finding by fitting and subtracting a polynomial of order 0 (offset) or 1 (offset + slope). "
				"The order of the polynomial is defined by contsub.order.");
			ImGui::PopStyleColor();

			ImGui::BeginDisabled(!model_.params.contsub.enable);

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning continuum subtraction order"))
			{
				static auto items = std::array{ "0", "1" };
				ImGui::PushID("order");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedCombo("Order", "##order", &model_.params.contsub.order, items.data(),
								static_cast<int>(items.size()));
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.contsub.order = 0;
				}
				ImGui::SetItemTooltip("Reset to default '0' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted("Order of the polynomial to be used in continuum subtraction if "
										   "contsub.enable = true. Can either "
										   "be 0 for a simple offset or 1 for an offset + slope. Higher orders are not "
										   "currently supported.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning continuum subtraction padding"))
			{
				ImGui::PushID("padding");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedDragInt("Padding", "##padding", &model_.params.contsub.padding, 1, 0, 1000 * 1000);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.contsub.padding = 3;
				}
				ImGui::SetItemTooltip("Reset to default '3' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"The amount of additional padding (in channels) applied to either side of channels "
						"excluded from the fit.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning continuum subtraction shift"))
			{
				ImGui::PushID("shift");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedDragInt("Shift", "##shift", &model_.params.contsub.shift, 1, 1, 1000 * 1000);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.contsub.shift = 4;
				}
				ImGui::SetItemTooltip("Reset to default '4' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"The number of channels by which the spectrum will be shifted (symmetrically in "
						"both directions) before self-subtraction.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning continuum subtraction threshold"))
			{
				ImGui::PushID("threshold");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedDragFloat("Threshold", "##threshold", &model_.params.contsub.threshold, 1, 0.0f,
									1000.0f * 1000.0f);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.contsub.threshold = 2.0f;
				}
				ImGui::SetItemTooltip("Reset to default '2.0f' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted("Relative clipping threshold. All channels with a flux density > "
										   "contsub.threshold times the noise "
										   "will be clipped and excluded from the polynomial fit.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			ImGui::EndDisabled();
			ImGui::PopStyleColor();
			ImGui::PopStyleVar();
		}
		ImGui::EndChild();
	}

	if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning flagging auto channels pixels"))
	{
		if (ImGui::BeginChild("##preconditioning_flagging", Vector2{ 0.0f, 0.0f },
							  ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY))
		{


			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

			ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
			ImGui::TextWrapped("Preconditioning Flagging");
			ImGui::PopFont();

			{
				const auto items = std::array{ "false", "true", "channels", "pixels" };
				ImGui::PushID("auto");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				static auto currentItem = 0;
				if (ui::HeadedCombo("Auto", "##auto", &currentItem, items.data(), static_cast<int>(items.size())))
				{
					model_.params.flag.autoMode = items[currentItem];
				}
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					currentItem = 0;
					model_.params.flag.autoMode = items[currentItem];
				}
				ImGui::SetItemTooltip("Reset to default 'false' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"If set to true, SoFiA will attempt to automatically flag spectral channels and spatial pixels "
						"affected by interference or artefacts based on their noise level. If set to channels, only "
						"spectral channels will be flagged. If set to pixels, only spatial pixels will be flagged. If "
						"set "
						"to false, auto-flagging will be disabled. Please see the user manual for details.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			ImGui::PopStyleColor();
			ImGui::PopStyleVar();
		}
		ImGui::EndChild();
	}


	if (not isFilterEnabled_ or
		paramsFilter_.PassFilter("preconditioning ripple filter grid xy z statistic window interpolate median mean"))
	{
		if (ImGui::BeginChild("##preconditioning_ripple_filter", Vector2{ 0.0f, 0.0f },
							  ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY))
		{

			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

			ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
			const auto wrapPosition = ImGui::GetContentRegionAvail().x - toggleSwitchWidth;
			ImGui::PushTextWrapPos(wrapPosition);

			const auto titleWidth = ImGui::CalcTextSize("Preconditioning Ripple Filter", 0, true, wrapPosition).x;
			ImGui::TextWrapped("Preconditioning Ripple Filter");
			ImGui::PopTextWrapPos();
			ImGui::PopFont();
			ImGui::SameLine(0.0f, wrapPosition - titleWidth);
			if (ui::ToggleSwitch(model_.params.ripple.enable, "##enable_preconditioning_ripple_filter", "off", "on"))
			{
				model_.params.ripple.enable = !model_.params.ripple.enable;
			}

			ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);
			ImGui::TextWrapped(
				"If enabled, the ripple filter will be applied to the data cube prior to source finding. "
				"The filter works by measuring and subtracting either the mean or median across a running window. "
				"This can be useful if a DC offset or spatial/spectral ripple is present in the data.");
			ImGui::PopStyleColor();

			ImGui::BeginDisabled(!model_.params.ripple.enable);

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning ripple filter grid xy"))
			{
				ImGui::PushID("grid xy");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				if (ui::HeadedDragInt("Grid XY", "##grid xy", &model_.params.ripple.gridXY, 2, 0, 1000 * 1000 + 1))
				{
					if (model_.params.ripple.gridXY > 0 and model_.params.ripple.gridXY % 2 == 0)
					{
						model_.params.ripple.gridXY += 1;
					}
				}
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.ripple.gridXY = 0;
				}
				ImGui::SetItemTooltip("Reset to default '0' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Spatial grid separation in pixels for the running window used by the background subtraction "
						"filter. The value must be an odd integer value and specifies the spatial step by which the "
						"window "
						"is moved. Alternatively, it can be set to 0 in which case it will default to half the spatial "
						"window size (see rippleFilter.windowXY).");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning ripple filter grid z"))
			{
				ImGui::PushID("grid z");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				if (ui::HeadedDragInt("Grid Z", "##grid z", &model_.params.ripple.gridZ, 2, 0, 1000 * 1000 + 1))
				{
					if (model_.params.ripple.gridZ > 0 and model_.params.ripple.gridZ % 2 == 0)
					{
						model_.params.ripple.gridZ += 1;
					}
				}
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.ripple.gridZ = 0;
				}
				ImGui::SetItemTooltip("Reset to default '0' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Spectral grid separation in channels for the running window used by the background "
						"subtraction "
						"filter. The value must be an odd integer value and specifies the spectral step by which the "
						"window is moved. Alternatively, it can be set to 0, in which case it will default to half the "
						"spectral window size (see rippleFilter.windowZ).");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning ripple filter interpolate"))
			{
				ImGui::PushID("interpolate");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedTextOnly("Interpolate");
				ImGui::Checkbox("##interpolate", &model_.params.ripple.interpolate);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.ripple.interpolate = false;
				}
				ImGui::SetItemTooltip("Reset to default 'false' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"If set to true then the mean or median values measured across the running "
						"window in the background subtraction filter will be linearly interpolated in "
						"between grid points. If set to false then the mean or median will be "
						"subtracted from the entire grid cell without interpolation.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning ripple filter statistic median mean"))
			{
				const auto items = std::array{ "median", "mean" };
				ImGui::PushID("statistic");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				static auto currentItem = 0;
				if (ui::HeadedCombo("Statistic", "##statistic", &currentItem, items.data(),
									static_cast<int>(items.size())))
				{
					model_.params.ripple.statistic = items[currentItem];
				}
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					currentItem = 0;
					model_.params.ripple.statistic = items[currentItem];
				}
				ImGui::SetItemTooltip("Reset to default 'median' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Controls whether the mean or median should be measured and subtracted in the "
						"running window of the background subtraction filter. The median is strongly "
						"recommended as it is more robust against real signal and artefacts.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning ripple filter window xy"))
			{
				ImGui::PushID("window xy");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedDragInt("Window XY", "##window xy", &model_.params.ripple.windowXY, 2, 1, 1000 * 1000 + 1);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.ripple.windowXY = 31;
				}
				ImGui::SetItemTooltip("Reset to default '31' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted("Spatial size in pixels of the running window used by the background "
										   "subtraction filter. The size must be an odd integer number.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning ripple filter window z"))
			{
				ImGui::PushID("window z");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedDragInt("Window Z", "##window z", &model_.params.ripple.windowZ, 2, 1, 1000 * 1000 + 1);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.ripple.windowZ = 15;
				}
				ImGui::SetItemTooltip("Reset to default '15' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted("Spectral size in channels of the running window used by the background "
										   "subtraction filter. The size must be an odd integer number.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			ImGui::EndDisabled();
			ImGui::PopStyleColor();
			ImGui::PopStyleVar();
		}
		ImGui::EndChild();
	}

	if (not isFilterEnabled_ or
		paramsFilter_.PassFilter("preconditioning noise scaling flux range positive negative full grid xy z "
								 "interpolate mode spectral local scfind statistic std mad gauss window"))
	{
		if (ImGui::BeginChild("##preconditioning_noise_scaling", Vector2{ 0.0f, 0.0f },
							  ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY))
		{

			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

			ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
			const auto wrapPosition = ImGui::GetContentRegionAvail().x - toggleSwitchWidth;
			ImGui::PushTextWrapPos(wrapPosition);

			const auto titleWidth = ImGui::CalcTextSize("Preconditioning Noise Scaling", 0, true, wrapPosition).x;
			ImGui::TextWrapped("Preconditioning Noise Scaling");
			ImGui::PopTextWrapPos();
			ImGui::PopFont();
			ImGui::SameLine(0.0f, wrapPosition - titleWidth);
			if (ui::ToggleSwitch(model_.params.scaleNoise.enable, "##enable_preconditioning_noise_scaling", "off",
								 "on"))
			{
				model_.params.scaleNoise.enable = !model_.params.scaleNoise.enable;
			}

			ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);
			ImGui::TextWrapped(
				"If enabled, noise scaling will be enabled. The purpose of the noise scaling modules is to measure the "
				"noise level in the input cube and then divide the input cube by the noise. This can be used to "
				"correct "
				"for spatial or spectral noise variations across the input cube prior to running the source finder.");
			ImGui::PopStyleColor();

			ImGui::BeginDisabled(!model_.params.scaleNoise.enable);


			if (not isFilterEnabled_ or
				paramsFilter_.PassFilter("preconditioning noise scaling flux range positive negative full"))
			{
				const auto items = std::array{ "negative", "positive", "full" };
				ImGui::PushID("flux_range");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				static auto currentItem = 0;
				if (ui::HeadedCombo("Flux Range", "##flux_range", &currentItem, items.data(),
									static_cast<int>(items.size())))
				{
					model_.params.scaleNoise.fluxRange = items[currentItem];
				}
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					currentItem = 0;
					model_.params.scaleNoise.fluxRange = items[currentItem];
				}
				ImGui::SetItemTooltip("Reset to default 'negative' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Flux density range to be used in the noise measurement. If set to negative or positive then "
						"only pixels with negative or positive flux density will be used, respectively. This can be "
						"helpful to prevent real emission or artefacts from affecting the noise measurement. If set to "
						"full then all pixels will be used in the noise measurement irrespective of their flux "
						"density.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning noise scaling grid xy"))
			{
				ImGui::PushID("grid xy");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				if (ui::HeadedDragInt("Grid XY", "##grid xy", &model_.params.scaleNoise.gridXY, 2, 0, 1000 * 1000 + 1))
				{
					if (model_.params.scaleNoise.gridXY > 0 and model_.params.scaleNoise.gridXY % 2 == 0)
					{
						model_.params.scaleNoise.gridXY += 1;
					}
				}
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.scaleNoise.gridXY = 0;
				}
				ImGui::SetItemTooltip("Reset to default '0' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted("Size of the spatial grid by which the noise measurement window is moved "
										   "across the data cube. It must be an odd integer value. If set to 0 then "
										   "the spatial grid size will default to half the spatial window size.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning noise scaling grid z"))
			{
				ImGui::PushID("grid z");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				if (ui::HeadedDragInt("Grid Z", "##grid z", &model_.params.scaleNoise.gridZ, 2, 0, 1000 * 1000 + 1))
				{
					if (model_.params.scaleNoise.gridZ > 0 and model_.params.scaleNoise.gridZ % 2 == 0)
					{
						model_.params.scaleNoise.gridZ += 1;
					}
				}
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.scaleNoise.gridZ = 0;
				}
				ImGui::SetItemTooltip("Reset to default '0' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted("Size of the spectral grid by which the noise measurement window is moved "
										   "across the data cube. It must be an odd integer value. If set to 0 then "
										   "the spectral grid size will default to half the spectral window size.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning noise scaling interpolate"))
			{
				ImGui::PushID("interpolate");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedTextOnly("Interpolate");
				ImGui::Checkbox("##interpolate", &model_.params.scaleNoise.interpolate);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.scaleNoise.interpolate = false;
				}
				ImGui::SetItemTooltip("Reset to default 'false' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"If set to true then linear interpolation will be used to interpolate the measured local noise "
						"values in between grid points. If set to false then the entire grid cell will instead be "
						"filled with the measured noise value.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning noise scaling mode spectral local"))
			{
				const auto items = std::array{ "spectral", "local" };
				ImGui::PushID("mode");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				static auto currentItem = 0;
				if (ui::HeadedCombo("Mode", "##mode", &currentItem, items.data(), static_cast<int>(items.size())))
				{
					model_.params.scaleNoise.mode = items[currentItem];
				}
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					currentItem = 0;
					model_.params.scaleNoise.mode = items[currentItem];
				}
				ImGui::SetItemTooltip("Reset to default 'spectral' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Noise scaling mode. If set to spectral then the noise level will be determined in each "
						"spectral channel by measuring the noise within each image plane. This is useful for data "
						"cubes where the noise varies with frequency. If set to local then the noise level will be "
						"measured locally in a window running across the entire data cube in all three dimensions. "
						"This is useful for data cubes with more complex spatial and spectral noise variations, e.g. "
						"interferometric data with primary-beam correction applied.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning noise scaling scfind"))
			{
				ImGui::PushID("scfind");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedTextOnly("Scfind");
				ImGui::Checkbox("##scfind", &model_.params.scaleNoise.scfind);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.scaleNoise.scfind = false;
				}
				ImGui::SetItemTooltip("Reset to default 'false' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"If true and global or local noise scaling is enabled then noise scaling will additionally be "
						"applied after each smoothing operation in the S+C finder. This might be useful in certain "
						"situations where large-scale artefacts are present in interferometric data. However, this "
						"feature should be used with great caution as it has the potential to do more harm than good.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or
				paramsFilter_.PassFilter("preconditioning noise scaling statistic std mad gauss"))
			{
				const auto items = std::array{ "mad", "std", "gauss" };
				ImGui::PushID("statistic");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				static auto currentItem = 0;
				if (ui::HeadedCombo("Statistic", "##statistic", &currentItem, items.data(),
									static_cast<int>(items.size())))
				{
					model_.params.scaleNoise.statistic = items[currentItem];
				}
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					currentItem = 0;
					model_.params.scaleNoise.statistic = items[currentItem];
				}
				ImGui::SetItemTooltip("Reset to default 'mad' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Statistic to be used in the noise measurement process. Possible values are std, mad and gauss "
						"for standard deviation, median absolute deviation and fitting of a Gaussian function to the "
						"flux histogram, respectively. Standard deviation is by far the fastest algorithm, but it is "
						"also the least robust with respect to emission and artefacts in the data. Median absolute "
						"deviation and Gaussian fitting are far more robust in the presence of strong, extended "
						"emission and artefacts, but will take slightly more time.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning noise scaling xy window"))
			{
				ImGui::PushID("window xy");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedDragInt("Window XY", "##window xy", &model_.params.ripple.windowXY, 2, 1, 1000 * 1000 + 1);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.ripple.windowXY = 25;
				}
				ImGui::SetItemTooltip("Reset to default '25' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Spatial size of the window used in determining the local noise level. It must be an odd "
						"integer value. If set to 0 then the pipeline will use the default value instead.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning noise scaling z window"))
			{
				ImGui::PushID("window z");
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedDragInt("Window Z", "##window z", &model_.params.ripple.windowZ, 2, 1, 1000 * 1000 + 1);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.ripple.windowZ = 15;
				}
				ImGui::SetItemTooltip("Reset to default '15' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Spectral size of the window used in determining the local noise level. It must be an odd "
						"integer value. If set to 0 then the pipeline will use the default value instead.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			ImGui::EndDisabled();
			ImGui::PopStyleColor();
			ImGui::PopStyleVar();
		}
		ImGui::EndChild();
	}

	if (not isFilterEnabled_ or
		paramsFilter_.PassFilter("source finding flux range positive negative full kernels xy z replacement statistic "
								 "std mad gauss threshold mode relative absolute"))
	{
		if (ImGui::BeginChild("##source_finding_settings", Vector2{ 0.0f, 0.0f },
							  ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY))
		{

			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

			ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
			ImGui::TextWrapped("Source Finding");
			ImGui::PopFont();

			ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);
			ImGui::TextWrapped(
				"The S+C finder operates by "
				"iteratively smoothing the data cube with a user-defined set of smoothing kernels, measuring the "
				"noise level on each smoothing scale, and adding all pixels with an absolute flux above a "
				"user-defined relative threshold to the source detection mask.");
			ImGui::PopStyleColor();

			{
				constexpr auto sourceFindingModeItems = std::array{ "S+C", "Threshold", "None" };
				static auto sourceFindingItem = 0;

				ImGui::PushID("##source_finding_mode");

				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				if (ui::HeadedCombo("Source Finding Method:", "##source_finding_mode", &sourceFindingItem,
									sourceFindingModeItems.data(),
									static_cast<uint32_t>(sourceFindingModeItems.size())))
				{
					model_.params.scfind.enable = false;
					model_.params.threshold.enable = false;
					if (sourceFindingItem == 0)
					{
						model_.params.scfind.enable = true;
					}
					else if (sourceFindingItem == 1)
					{
						model_.params.threshold.enable = true;
					}
				}

				ImGui::PopID();
			}

			ImGui::BeginDisabled(!model_.params.scfind.enable);

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("source finding flux range positive negative full"))
			{
				constexpr auto fluxRangeItems = std::array{ "negative", "positive", "full" };
				static auto fluxRangeItem = 0;

				ImGui::PushID("##scfind_flux_range");

				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				if (ui::HeadedCombo("Flux Range:", "##scfind_flux_range", &fluxRangeItem, fluxRangeItems.data(),
									static_cast<uint32_t>(fluxRangeItems.size())))
				{
					model_.params.scfind.fluxRange = fluxRangeItems[fluxRangeItem];
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal) and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Flux range to be used in the S+C finder noise measurement. If set to negative or positive "
						"then only pixels with negative or positive flux density will be used, respectively. This can "
						"be useful to prevent real emission or artefacts from affecting the noise measurement. If set "
						"to full then all pixels will be used in the noise measurement irrespective of their flux.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);

				if (ui::Button(ICON_LC_UNDO_2))
				{
					fluxRangeItem = 0;
					model_.params.scfind.fluxRange = fluxRangeItems[fluxRangeItem];
				}
				ImGui::SetItemTooltip("Reset to default 'negative' value");
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("source finding kernels xy"))
			{
				ImGui::PushID("scfind_kernels_xy");
				ui::HeadedTextOnly("XY Kernels:");

				const auto buttonHeight = style.FramePadding.y * 2.0f + ImGui::CalcTextSize("0").y;
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, Vector2{ 16.0f, 16.0f });
				const auto itemHeight = style.FramePadding.y * 2.0f + buttonHeight + style.ScrollbarSize;

				ImGui::BeginChild(
					"##kernel_selection_xy",
					Vector2{ ImGui::GetContentRegionAvail().x - undoButtonDefaultSize - 8.0f, itemHeight },
					ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_FrameStyle,
					ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysHorizontalScrollbar);
				ImGui::PopStyleVar();

				for (auto i : model_.params.scfind.kernelsXY)
				{
					selectedKernelsXY_.insert(i);
				}
				static auto kernelsXY = 30;
				for (auto i = 0; i < kernelsXY; i++)
				{
					ImGui::PushID(i);
					if (ui::ToggleButton(selectedKernelsXY_.contains(i), std::format("{}##b", i).c_str()))
					{
						if (selectedKernelsXY_.contains(i))
						{
							selectedKernelsXY_.erase(i);
							model_.params.scfind.kernelsXY.erase(std::find(model_.params.scfind.kernelsXY.begin(),
																		   model_.params.scfind.kernelsXY.end(), i));
						}
						else
						{
							selectedKernelsXY_.insert(i);
							model_.params.scfind.kernelsXY.push_back(i);
						}
					}
					ImGui::PopID();
					ImGui::SameLine();
				}
				if (ui::AccentButton("Add##add_more_filter"))
				{
					kernelsXY++;
					ImGui::SameLine(0.0f, style.FramePadding.x);
					ImGui::Dummy(ImGui::GetItemRectSize());
					ImGui::SetScrollHereX(0.0);
				}
				ImGui::EndChild();
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);

				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.scfind.kernelsXY = { 0, 3, 6 };
					selectedKernelsXY_.clear();
				}
				ImGui::SetItemTooltip("Reset to default XY kernel sizes '0, 3, 6'");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Comma-separated list of spatial Gaussian kernel sizes to apply. The individual kernel sizes "
						"must "
						"be "
						"floating-point values and denote the full width at half maximum (FWHM) of the Gaussian used "
						"to "
						"smooth "
						"the "
						"data in the spatial domain. A value of 0 means that no spatial smoothing will be applied.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("source finding kernels z"))
			{
				ImGui::PushID("scfind_kernels_z");
				ui::HeadedTextOnly("Z Kernels:");

				const auto buttonHeight = style.FramePadding.y * 2.0f + ImGui::CalcTextSize("0").y;
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, Vector2{ 16.0f, 16.0f });
				const auto itemHeight = style.FramePadding.y * 2.0f + buttonHeight + style.ScrollbarSize;

				ImGui::BeginChild(
					"##kernel_selection_z",
					Vector2{ ImGui::GetContentRegionAvail().x - undoButtonDefaultSize - 8.0f, itemHeight },
					ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_FrameStyle,
					ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysHorizontalScrollbar);
				ImGui::PopStyleVar();

				for (auto i : model_.params.scfind.kernelsZ)
				{
					selectedKernelsZ_.insert(i);
				}
				static auto kernelsZ = 60;
				for (auto i = 0; i < kernelsZ; i += 2)
				{
					ImGui::PushID(i);
					if (ui::ToggleButton(selectedKernelsZ_.contains(i), std::format("{}##b", i).c_str()))
					{
						if (selectedKernelsZ_.contains(i))
						{
							selectedKernelsZ_.erase(i);
							model_.params.scfind.kernelsZ.erase(std::find(model_.params.scfind.kernelsZ.begin(),
																		  model_.params.scfind.kernelsZ.end(), i));
						}
						else
						{
							selectedKernelsZ_.insert(i);
							model_.params.scfind.kernelsZ.push_back(i);
						}
					}
					ImGui::PopID();
					ImGui::SameLine();
					if (i == 0)
					{
						i++;
					}
				}
				if (ui::AccentButton("Add##add_more_filter"))
				{
					kernelsZ += 2;
					ImGui::SameLine(0.0f, style.FramePadding.x);
					ImGui::Dummy(ImGui::GetItemRectSize());
					ImGui::SetScrollHereX(0.0);
				}
				ImGui::EndChild();
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);

				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.scfind.kernelsZ = { 0, 3, 7, 15 };
					selectedKernelsZ_.clear();
				}
				ImGui::SetItemTooltip("Reset to default Z kernel sizes '0, 3, 7, 15'");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted("Comma-separated list of spectral Boxcar kernel sizes to apply. The "
										   "individual kernel sizes must "
										   "be "
										   "odd integer values of 3 or greater and denote the full width of the Boxcar "
										   "filter used to smooth "
										   "the data in the spectral domain. A value of 0 means that no spectral "
										   "smoothing will be applied.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}

				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("source finding replacement"))
			{
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedDragFloat("Replacement:", "##scfind_replacement", &model_.params.scfind.replacement, 0.01f,
									-1.0f, 9999.0f);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2))
				{
					model_.params.scfind.replacement = 2.0f;
				}
				ImGui::SetItemTooltip("Reset to default '2.0' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Before smoothing the data cube during an S+C iteration, every pixel in the data cube that was "
						"already "
						"detected in a previous iteration will be replaced by this value multiplied by the original "
						"noise "
						"level in "
						"the non-smoothed data cube, while keeping the original sign of the data value. This feature "
						"can "
						"be "
						"disabled altogether by specifying a value of < 0.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("source finding statistic std mad gauss threshold"))
			{
				constexpr auto scaleNoiseStatisticItems = std::array{ "std", "mad", "gauss" };
				static auto scaleNoiseStatisticItem = 1;

				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				if (ui::HeadedCombo("Statistic:", "##scfind_statistic", &scaleNoiseStatisticItem,
									scaleNoiseStatisticItems.data(),
									static_cast<uint32_t>(scaleNoiseStatisticItems.size())))
				{
					model_.params.scfind.statistic = scaleNoiseStatisticItems[scaleNoiseStatisticItem];
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal) and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Statistic to be used in the noise measurement process. Possible values are std, mad and gauss "
						"for "
						"standard deviation, median absolute deviation and Gaussian fitting to the flux histogram, "
						"respectively. "
						"Standard deviation is by far the fastest algorithm, but it is also the least robust one with "
						"respect "
						"to "
						"emission and artefacts in the data. Median absolute deviation and Gaussian fitting are far "
						"more "
						"robust in "
						"the presence of strong, extended emission or artefacts, but will usually take longer.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2 "##scfind_statistic"))
				{
					scaleNoiseStatisticItem = 1;
					model_.params.scfind.statistic = scaleNoiseStatisticItems[scaleNoiseStatisticItem];
				}
				ImGui::SetItemTooltip("Reset to default 'mad' value");
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("source finding threshold"))
			{
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedDragFloat("Threshold:", "##scfind_threshold", &model_.params.scfind.threshold, 0.01f, 0.0f,
									100.0f);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2 "##scfind_threshold"))
				{
					model_.params.scfind.threshold = 5.0f;
				}
				ImGui::SetItemTooltip("Reset to default '5.0' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Flux threshold to be used by the S+C finder relative to the measured noise level in each "
						"smoothing iteration. In practice, values in the range of about 3 to 5 have proven to be "
						"useful in most situations, with lower values in that range requiring use of the "
						"reliability filter to reduce the number of false detections.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
			}
			ImGui::EndDisabled();


			ImGui::BeginDisabled(!model_.params.threshold.enable);

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("source finding flux range positive negative full"))
			{
				constexpr auto fluxRangeItems = std::array{ "negative", "positive", "full" };
				static auto fluxRangeItem = 0;

				ImGui::PushID("##threshold_flux_range");

				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				if (ui::HeadedCombo("Flux Range:", "##threshold_flux_range", &fluxRangeItem, fluxRangeItems.data(),
									static_cast<uint32_t>(fluxRangeItems.size())))
				{
					model_.params.scfind.fluxRange = fluxRangeItems[fluxRangeItem];
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal) and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Flux range to be used in the threshold finder noise measurement. If set to negative or "
						"positive "
						"then only pixels with negative or positive flux density will be used, respectively. This can "
						"be "
						"helpful to prevent real emission or artefacts from affecting the noise measurement. If set to "
						"full then all pixels will be used in the noise measurement irrespective of their flux.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);

				if (ui::Button(ICON_LC_UNDO_2))
				{
					fluxRangeItem = 0;
					model_.params.scfind.fluxRange = fluxRangeItems[fluxRangeItem];
				}
				ImGui::SetItemTooltip("Reset to default 'negative' value");
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("source finding threshold mode relative absolute"))
			{
				constexpr auto thresholdModes = std::array{ "relative", "absolute" };
				static auto thresholdModeItem = 0;

				ImGui::PushID("##threshold_mode");

				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				if (ui::HeadedCombo("Mode:", "##threshold_mode", &thresholdModeItem, thresholdModes.data(),
									static_cast<uint32_t>(thresholdModes.size())))
				{
					model_.params.threshold.mode = thresholdModes[thresholdModeItem];
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal) and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"If set to absolute then the flux threshold of the threshold finder will be interpreted as an "
						"absolute flux density threshold in the native flux unit of the input data cube. If set to "
						"relative then the threshold will be relative to the noise level in the input data cube.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);

				if (ui::Button(ICON_LC_UNDO_2))
				{
					thresholdModeItem = 0;
					model_.params.threshold.mode = thresholdModes[thresholdModeItem];
				}
				ImGui::SetItemTooltip("Reset to default 'relative' value");
				ImGui::PopID();
			}

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("source finding statistic std mad gauss threshold"))
			{
				constexpr auto statisticItems = std::array{ "std", "mad", "gauss" };
				static auto statisticItem = 1;

				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				if (ui::HeadedCombo("Statistic:", "##threshold_statistic", &statisticItem, statisticItems.data(),
									static_cast<uint32_t>(statisticItems.size())))
				{
					model_.params.threshold.statistic = statisticItems[statisticItem];
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal) and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted(
						"Statistic to be used in the noise measurement process if threshold.mode is set to relative. "
						"Possible values are std, mad and gauss for standard deviation, median absolute deviation and "
						"fitting of a Gaussian function to the flux histogram, respectively. Standard deviation is by "
						"far "
						"the fastest algorithm, but it is also the least robust with respect to emission and artefacts "
						"in "
						"the data. Median absolute deviation and Gaussian fitting are far more robust in the presence "
						"of "
						"strong, extended emission or artefacts, but will usually take slightly more time.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2 "##threshold_statistic"))
				{
					statisticItem = 1;
					model_.params.threshold.statistic = statisticItems[statisticItem];
				}
				ImGui::SetItemTooltip("Reset to default 'mad' value");
			}
			if (not isFilterEnabled_ or paramsFilter_.PassFilter("source finding threshold"))
			{
				ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
				ui::HeadedDragFloat("Threshold:", "##threshold_threshold", &model_.params.threshold.threshold, 0.01f,
									0.0f, 100.0f);
				const auto isHovered = ImGui::IsItemHovered(
					ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_ForTooltip);
				ImGui::PopItemWidth();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button(ICON_LC_UNDO_2 "##threshold_threshold"))
				{
					model_.params.threshold.threshold = 5.0f;
				}
				ImGui::SetItemTooltip("Reset to default '5.0' value");

				if (isHovered and ImGui::BeginTooltip())
				{
					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
					ImGui::TextUnformatted("Flux threshold to be applied by the threshold finder. Depending on the "
										   "threshold.mode parameter, this can either be absolute (in native flux "
										   "units of the input data cube) or relative to the noise level of the cube.");
					ImGui::PopTextWrapPos();
					ImGui::EndTooltip();
				}
			}

			ImGui::EndDisabled();
			ImGui::PopStyleVar();
			ImGui::PopStyleColor();
		}


		ImGui::EndChild();
	}

	if (not isFilterEnabled_ or
		paramsFilter_.PassFilter("linking linker keep negative max fill pixels size xy z min positivity radius"))
	{
		ImGui::BeginChild("##linking", Vector2{ 0.0f, 0.0f }, ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY);
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

		ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
		const auto wrapPosition = ImGui::GetContentRegionAvail().x - toggleSwitchWidth;
		ImGui::PushTextWrapPos(wrapPosition);

		const auto titleWidth = ImGui::CalcTextSize("Linking", 0, true, wrapPosition).x;
		ImGui::TextWrapped("Linking");
		ImGui::PopTextWrapPos();
		ImGui::PopFont();
		ImGui::SameLine(0.0f, wrapPosition - titleWidth);
		if (ui::ToggleSwitch(model_.params.linker.enable, "##enable_linking", "off", "on"))
		{
			model_.params.linker.enable = !model_.params.linker.enable;
		}

		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);
		ImGui::TextWrapped(
			"If enabled, then the linker will be run to merge the pixels detected by the source finder into "
			"coherent detections that can then be parameterised and catalogued. If false, the pipeline will be "
			"terminated after source finding, and no catalogue or source products will be created. Disabling "
			"the linker can be useful if only the raw mask from the source finder is needed.");
		ImGui::PopStyleColor();


		const auto isSubcube = isUsingSubcube();

		ImGui::BeginDisabled(!model_.params.linker.enable or isSubcube);

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker keep negative"))
		{
			ImGui::PushID("linker_keep_negative");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedTextOnly("Keep Negative");
			ImGui::Checkbox("##linker_keep_negative", &model_.params.linker.keepNegative);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.keepNegative = false;
			}
			ImGui::SetItemTooltip("Reset to default 'false' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"If set to true then the linker will not discard detections with negative flux. Reliability "
					"filtering must be disabled for negative sources to be retained. Also note that negative sources "
					"will not appear in moment 1 and 2 maps. This option should only ever be used for testing or "
					"debugging purposes, but never in production mode.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker min fill"))
		{
			ImGui::PushID("linker_min_fill");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragFloat("Min Fill:", "##linker_min_fill", &model_.params.linker.minFill, 0.01f, 0.0f, 9999.0f);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.minFill = 0.0f;
			}
			ImGui::SetItemTooltip("Reset to default '0.0' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"Minimum allowed filling factor of a source within its rectangular bounding box, defined as the "
					"number of spatial and spectral pixels that make up the source divided by the number of pixels in "
					"the bounding box. The default value of 0.0 disables minimum filling factor filtering.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker max fill"))
		{
			ImGui::PushID("linker_max_fill");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragFloat("Max Fill:", "##linker_max_fill", &model_.params.linker.maxFill, 0.01f, 0.0f, 9999.0f);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.maxFill = 0.0f;
			}
			ImGui::SetItemTooltip("Reset to default '0.0' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"Maximum allowed filling factor of a source within its rectangular bounding box, defined as the "
					"number of spatial and spectral pixels that make up the source divided by the number of pixels in "
					"the bounding box. The default value of 0.0 disables maximum filling factor filtering.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker min pixels"))
		{
			ImGui::PushID("linker_min_pixels");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Min Pixels:", "##linker_min_pixels", &model_.params.linker.minPixels, 1.0f, 0,
							  1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.minPixels = 0;
			}
			ImGui::SetItemTooltip("Reset to default '0' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted("Minimum allowed number of spatial and spectral pixels that a source must have. "
									   "The default value of 0 disables minimum size filtering.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker max pixels"))
		{
			ImGui::PushID("linker_max_pixels");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Max Pixels:", "##linker_max_pixels", &model_.params.linker.maxPixels, 1.0f, 0,
							  1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.maxPixels = 0;
			}
			ImGui::SetItemTooltip("Reset to default '0' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted("Maximum allowed number of spatial and spectral pixels that a source must not "
									   "exceed. The default value of 0 disables maximum size filtering.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker min size xy"))
		{
			ImGui::PushID("linker_min_size_xy");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Min Size XY:", "##linker_min_size_xy", &model_.params.linker.minSizeXY, 1.0f, 0,
							  1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.minSizeXY = 0;
			}
			ImGui::SetItemTooltip("Reset to default '0' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted("Minimum size of sources in the spatial dimension in pixels. Sources that fall "
									   "below this limit will be discarded by the linker.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker max size xy"))
		{
			ImGui::PushID("linker_max_size_xy");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Max Size XY:", "##linker_max_size_xy", &model_.params.linker.maxSizeXY, 1.0f, 0,
							  1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.maxSizeXY = 0;
			}
			ImGui::SetItemTooltip("Reset to default '0' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"Maximum size of sources in the spatial dimension in pixels. Sources that exceed this limit will "
					"be discarded by the linker. If the value is set to 0, maximum size filtering will be disabled.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker min size z"))
		{
			ImGui::PushID("linker_min_size_z");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Min Size Z:", "##linker_min_size_z", &model_.params.linker.minSizeZ, 1.0f, 1,
							  1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.minSizeZ = 5;
			}
			ImGui::SetItemTooltip("Reset to default '5' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted("Minimum size of sources in the spectral dimension in pixels. Sources that fall "
									   "below this limit will be discarded by the linker.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker max size z"))
		{
			ImGui::PushID("linker_max_size_z");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Max Size Z:", "##linker_max_size_z", &model_.params.linker.maxSizeZ, 1.0f, 0,
							  1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.minSizeZ = 0;
			}
			ImGui::SetItemTooltip("Reset to default '0' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"Maximum size of sources in the spectral dimension in pixels. Sources that exceed this limit will "
					"be discarded by the linker. If the value is set to 0, maximum size filtering will be disabled.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker positivity"))
		{
			ImGui::PushID("linker_positivity");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedTextOnly("Positivity:");
			ImGui::Checkbox("##linker_positivity", &model_.params.linker.positivity);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.positivity = false;
			}
			ImGui::SetItemTooltip("Reset to default 'false' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"If set to true then the linker will only merge positive pixels and discard all negative pixels by "
					"removing them from the mask. This option should be used with extreme caution and will render the "
					"reliability filter useless. It can be useful, though, if there are significant negative artefacts "
					"such as residual sidelobes in the data.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker xy radius"))
		{
			ImGui::PushID("linker_radius_xy");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Radius XY:", "##linker_radius_xy", &model_.params.linker.radiusXY, 1.0f, 1, 1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.radiusXY = 1;
			}
			ImGui::SetItemTooltip("Reset to default '1' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted("Maximum merging length in the spatial dimension. Pixels with a separation of "
									   "up to this value will be merged into the same source.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("linking linker z radius"))
		{
			ImGui::PushID("linker_radius_z");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Radius Z:", "##linker_radius_z", &model_.params.linker.radiusZ, 1.0f, 1, 1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.linker.radiusZ = 1;
			}
			ImGui::SetItemTooltip("Reset to default '1' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted("Maximum merging length in the spectral dimension. Pixels with a separation of "
									   "up to this value will be merged into the same source.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		ImGui::EndDisabled();
		ImGui::PopStyleColor();
		ImGui::PopStyleVar();
		ImGui::EndChild();
	}

	if (not isFilterEnabled_ or
		paramsFilter_.PassFilter("reliability auto kernel catalog debug iterations min pixels min snr parameters peak "
								 "sum mean chan pix fill std skew kurt plot extra scale kernel threshold tolerance"))
	{
		ImGui::BeginChild("##reliability", Vector2{ 0.0f, 0.0f },
						  ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY);
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

		ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
		const auto wrapPosition = ImGui::GetContentRegionAvail().x - toggleSwitchWidth;
		ImGui::PushTextWrapPos(wrapPosition);

		const auto titleWidth = ImGui::CalcTextSize("Reliability", 0, true, wrapPosition).x;
		ImGui::TextWrapped("Reliability");
		ImGui::PopTextWrapPos();
		ImGui::PopFont();
		ImGui::SameLine(0.0f, wrapPosition - titleWidth);
		if (ui::ToggleSwitch(model_.params.reliability.enable, "##enable_reliability", "off", "on"))
		{
			model_.params.reliability.enable = !model_.params.reliability.enable;
		}

		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);
		ImGui::TextWrapped(
			"If enabled, reliability calculation and filtering will be enabled. This will determine the "
			"reliability of each detection with positive total flux by comparing the density of positive and "
			"negative detections in a three-dimensional parameter space. Sources below the specified "
			"reliability threshold will then be discarded. Note that this will require a sufficient number of "
			"negative detections, which can usually be achieved by setting the source finding threshold to "
			"somewhere around 3 to 4 times the noise level.");
		ImGui::PopStyleColor();

		const auto isSubcube = isUsingSubcube();

		ImGui::BeginDisabled(!model_.params.reliability.enable or isSubcube);


		if (not isFilterEnabled_ or paramsFilter_.PassFilter("reliability auto kernel"))
		{
			ImGui::PushID("auto_kernel");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedTextOnly("Auto Kernel:");
			ImGui::Checkbox("##auto_kernel", &model_.params.reliability.autoKernel);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.reliability.autoKernel = false;
			}
			ImGui::SetItemTooltip("Reset to default 'false' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"If set to true then SoFiA will try to automatically determine the optimal reliability kernel "
					"scale factor by iteratively increasing the kernel size until the absolute value of the median of "
					"the Skellam distribution decreases below reliability.tolerance. If the algorithm fails to "
					"converge after reliability.iterations steps then the default value of reliability.scaleKernel "
					"will be used instead.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("reliability iterations"))
		{
			ImGui::PushID("reliability_iterations");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Iterations:", "##reliability_iterations", &model_.params.reliability.iterations, 1, 1,
							  1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.reliability.iterations = 30;
			}
			ImGui::SetItemTooltip("Reset to default '30' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted("Maximum number of iterations of the reliability kernel auto-scaling algorithm. "
									   "If convergence is not achieved within this number of iterations then a scaling "
									   "factor of reliability.scaleKernel will instead be applied.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("reliability min pixels"))
		{
			ImGui::PushID("reliability_min_pixels");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Min Pixels:", "##reliability_min_pixels", &model_.params.reliability.minPixels, 1, 0,
							  1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.reliability.minPixels = 0;
			}
			ImGui::SetItemTooltip("Reset to default '0' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted("Minimum total number of spatial and spectral pixels within the source mask for "
									   "detections to be considered reliable. The reliability of any detection with "
									   "fewer pixels will be set to zero by default.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("reliability min snr"))
		{
			ImGui::PushID("reliability_min_snr");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragFloat("Min SNR:", "##reliability_min_snr", &model_.params.reliability.minSNR, 1, 0,
								1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.reliability.minSNR = 3.0f;
			}
			ImGui::SetItemTooltip("Reset to default '3.0' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"Lower signal-to-noise limit for reliable sources. Detections that fall below this threshold will "
					"be deemed unreliable and assigned a reliability of 0. The value denotes the integrated "
					"signal-to-noise ratio, SNR = F_sum / [RMS * sqrt(N * Omega)], of the source, where Omega is the "
					"solid angle (in pixels) of the point spread function of the data, N is the number of spatial and "
					"spectral pixels of the source, F_sum is the summed flux density and RMS is the local RMS noise "
					"level (assumed to be constant). Note that the spectral resolution is assumed to be equal to the "
					"channel width. If BMAJ and BMIN are not defined in the input FITS file header then Omega will be "
					"set to 1 by default, thus assuming a beam size of 1 pixel.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or
			paramsFilter_.PassFilter("reliability parameters peak sum mean chan pix fill std skew kurt"))
		{
			const auto parameters = std::array{ "peak", "sum", "mean", "chan", "pix", "fill", "std", "skew", "kurt" };
			ImGui::PushID("reliability_parameters");
			ui::HeadedTextOnly("Parameters:");

			const auto buttonHeight = style.FramePadding.y * 2.0f + ImGui::CalcTextSize("0").y;
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, Vector2{ 16.0f, 16.0f });
			const auto itemHeight = style.FramePadding.y * 2.0f + buttonHeight + style.ScrollbarSize;

			ImGui::BeginChild("##reliability_parameters",
							  Vector2{ ImGui::GetContentRegionAvail().x - undoButtonDefaultSize - 8.0f, itemHeight },
							  ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_FrameStyle,
							  ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysHorizontalScrollbar);
			ImGui::PopStyleVar();


			for (auto i = 0; i < parameters.size(); i++)
			{
				ImGui::PushID(i);

				const auto isSelected =
					std::find(model_.params.reliability.parameters.begin(), model_.params.reliability.parameters.end(),
							  parameters[i]) != model_.params.reliability.parameters.end();
				if (ui::ToggleButton(isSelected, std::format("{}##b", parameters[i]).c_str()))
				{
					if (isSelected)
					{
						model_.params.reliability.parameters.erase(
							std::find(model_.params.reliability.parameters.begin(),
									  model_.params.reliability.parameters.end(), parameters[i]));
					}
					else
					{
						model_.params.reliability.parameters.push_back(parameters[i]);
					}
				}
				ImGui::PopID();
				ImGui::SameLine();
			}

			ImGui::EndChild();
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);

			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.reliability.parameters.clear();
				model_.params.reliability.parameters.push_back("peak");
				model_.params.reliability.parameters.push_back("sum");
				model_.params.reliability.parameters.push_back("mean");
			}
			ImGui::SetItemTooltip("Reset to default parameters: peak, sum, mean");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"Parameter space to be used in deriving the reliability of detections. This must be a list of "
					"parameters the number of which defines the dimensionality of the parameter space. Possible values "
					"are peak for the peak flux density, sum for the summed flux density, mean for mean flux density, "
					"chan for the number of spectral channels, pix for the total number of spatial and spectral "
					"pixels, fill for the filling factor, std for the standard deviation, skew for the skewness and "
					"kurt for the kurtosis across the source mask. Flux densities will be divided by the global noise "
					"level. peak, sum, mean, pix and fill will be logarithmic, all other parameters linear.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}

			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("reliability scale kernel"))
		{
			ImGui::PushID("reliability_scale_kernel");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragFloat("Scale Kernel:", "##reliability_scale_kernel", &model_.params.reliability.scaleKernel,
								0.001f, 0, 1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.reliability.scaleKernel = 0.4f;
			}
			ImGui::SetItemTooltip("Reset to default '0.4' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"When estimating the density of positive and negative detections in parameter space, the size of "
					"the Gaussian kernel used in this process is determined from the covariance of the distribution of "
					"negative detections in parameter space. This parameter setting can be used to scale that kernel "
					"size by a constant factor.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("reliability threshold"))
		{
			ImGui::PushID("reliability_threshold");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragFloat("Threshold:", "##reliability_threshold", &model_.params.reliability.threshold, 0.001f,
								0.0f, 1.0f);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.reliability.threshold = 0.9f;
			}
			ImGui::SetItemTooltip("Reset to default '0.9' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted("Reliability threshold in the range of 0 to 1. Sources with a reliability below "
									   "this threshold will be discarded.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("reliability tolerance"))
		{
			ImGui::PushID("reliability_tolerance");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragFloat("Tolerance:", "##reliability_tolerance", &model_.params.reliability.tolerance, 0.001f,
								0.0f, 1000.0f * 1000.0f);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.reliability.tolerance = 0.05f;
			}
			ImGui::SetItemTooltip("Reset to default '0.05' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"Convergence tolerance for the reliability kernel auto-scaling algorithm. Convergence is achieved "
					"when the absolute value of the median of the Skellam distribution drops below this tolerance.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		ImGui::EndDisabled();
		ImGui::PopStyleColor();
		ImGui::PopStyleVar();
		ImGui::EndChild();
	}

	if (not isFilterEnabled_ or paramsFilter_.PassFilter("mask dilation iterations xy z threshold"))
	{
		ImGui::BeginChild("##mask_dilation", Vector2{ 0.0f, 0.0f },
						  ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY);
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

		ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
		const auto wrapPosition = ImGui::GetContentRegionAvail().x - toggleSwitchWidth;
		ImGui::PushTextWrapPos(wrapPosition);

		const auto titleWidth = ImGui::CalcTextSize("Mask Dilation", 0, true, wrapPosition).x;
		ImGui::TextWrapped("Mask Dilation:");
		ImGui::PopTextWrapPos();
		ImGui::PopFont();
		ImGui::SameLine(0.0f, wrapPosition - titleWidth);
		if (ui::ToggleSwitch(model_.params.dilation.enable, "##enable_mask_dilation", "off", "on"))
		{
			model_.params.dilation.enable = !model_.params.dilation.enable;
		}

		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);
		ImGui::TextWrapped(
			"Set to true to enable source mask dilation whereby the mask of each source will be grown outwards "
			"until the resulting increase in integrated flux drops below a given threshold or the maximum "
			"number of iterations is reached.");
		ImGui::PopStyleColor();

		const auto isSubcube = isUsingSubcube();

		ImGui::BeginDisabled(!model_.params.dilation.enable or isSubcube);

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("mask dilation iterations xy"))
		{
			ImGui::PushID("dilation_iterations_xy");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Iterations XY:", "##dilation_iterations_xy", &model_.params.dilation.iterationsXY, 1, 1,
							  1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.dilation.iterationsXY = 10;
			}
			ImGui::SetItemTooltip("Reset to default '10' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"Sets the maximum number of spatial iterations for the mask dilation algorithm. Once this number "
					"of iterations has been reached, mask dilation in the spatial plane will stop even if the flux "
					"increase still exceeds the threshold set by dilation.threshold.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("mask dilation iterations z"))
		{
			ImGui::PushID("dilation_iterations_z");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragInt("Iterations Z:", "##dilation_iterations_z", &model_.params.dilation.iterationsZ, 1, 1,
							  1000 * 1000);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.dilation.iterationsZ = 5;
			}
			ImGui::SetItemTooltip("Reset to default '5' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"Sets the maximum number of spectral iterations for the mask dilation algorithm. Once this number "
					"of iterations has been reached, mask dilation along the spectral axis will stop even if the flux "
					"increase still exceeds the threshold set by dilation.threshold.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		if (not isFilterEnabled_ or paramsFilter_.PassFilter("mask dilation threshold"))
		{
			ImGui::PushID("dilation_threshold");
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 8.0f - undoButtonDefaultSize);
			ui::HeadedDragFloat("Threshold:", "##dilation_threshold", &model_.params.dilation.threshold, 0.0001f, 0.0f,
								1000.0f * 1000.0f);
			const auto isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_Stationary |
														ImGuiHoveredFlags_ForTooltip);
			ImGui::PopItemWidth();
			ImGui::SameLine(0.0f, 8.0f);
			if (ui::Button(ICON_LC_UNDO_2))
			{
				model_.params.dilation.threshold = 0.001f;
			}
			ImGui::SetItemTooltip("Reset to default '0.001' value");

			if (isHovered and ImGui::BeginTooltip())
			{
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
					"If a positive value is provided, mask dilation will end when the increment in the integrated flux "
					"during a single iteration drops below this value times the total integrated flux (from the "
					"previous iteration), or when the maximum number of iterations has been reached. Specifying a "
					"negative threshold will disable flux checking altogether and always carry out the maximum number "
					"of iterations.");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}
			ImGui::PopID();
		}

		ImGui::EndDisabled();
		ImGui::PopStyleColor();
		ImGui::PopStyleVar();
		ImGui::EndChild();
	}

	ImGui::PopStyleVar();

	ImGui::EndDisabled();
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

auto SoFiaSearchView::isUsingSubcube() -> bool
{
	const auto& brush = ApplicationContext::getStyleBrush();
	const auto hasProject = applicationContext_->selectedProject_.has_value();

	auto dimensions = owl::vec3i{ 0 };
	if (hasProject)
	{
		const auto& dims = applicationContext_->selectedProject_.value().fitsOriginProperties.axisDimensions;
		dimensions = { dims[0], dims[1], dims[2] };
	}

	auto isSubcube = model_.showRoiGizmo or model_.params.input.region.lower.x != 0 or
		model_.params.input.region.lower.y != 0 or model_.params.input.region.lower.z != 0 or
		model_.params.input.region.upper.x != dimensions[0] or model_.params.input.region.upper.y != dimensions[1] or
		model_.params.input.region.upper.z != dimensions[2];

	if (isSubcube)
	{
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.systemFillColorCautionBackgroundBrush);

		ImGui::BeginChild("##project_not_selected_warning", Vector2{},
						  ImGuiChildFlags_AlwaysAutoResize | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_FrameStyle);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.systemFillColorCautionBrush);
		ImGui::Text(ICON_LC_MESSAGE_SQUARE_WARNING);
		ImGui::PopStyleColor();
		ImGui::SameLine();
		if (ui::Button("Resolve##reset_region"))
		{
			model_.showRoiGizmo = false;
			model_.transform = {};
			model_.params.input.region.lower.x = 0;
			model_.params.input.region.lower.y = 0;
			model_.params.input.region.lower.z = 0;
			model_.params.input.region.upper.x = dimensions[0];
			model_.params.input.region.upper.y = dimensions[1];
			model_.params.input.region.upper.z = dimensions[2];
		}
		ImGui::SetItemTooltip("Disables sub region input!");
		ImGui::SameLine();
		const auto text = "This stage will be skipped by SoFiA because of the sub region selection. Consider "
						  "reseting or disabling the sub region to the original size!";
		ImGui::TextWrapped(text);
		ImGui::EndChild();
		ImGui::PopStyleColor();
	}

	return hasProject and isSubcube;
}
