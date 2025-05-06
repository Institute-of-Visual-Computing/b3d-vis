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
	ImGui::TextLinkOpenURL("Learn more about SoFiA 2 control parameters", "https://gitlab.com/SoFiA-Admin/SoFiA-2/-/wikis/SoFiA-2-Control-Parameters");
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
	const auto filterChanged = ui::InputText("##sofia_params_filter", "SoFiA Parameters Filter", paramsFilter_.InputBuf,
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

	/*ImGui::Begin("teststst");
	drawFilterFormContentOld();
	ImGui::End();*/
}

auto SoFiaSearchView::drawFilterFormContent() -> void
{
	const auto hasProject = applicationContext_->selectedProject_.has_value();

	constexpr auto toggleSwitchWidth = 14.0f * 4.0f + 4.0f;
	const auto disableInteraction = false;

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
		paramsFilter_.PassFilter("preconditioning continuum substraction order padding shift threshhold"))
	{
		if (ImGui::BeginChild("##preconditioning_continuum_substraction", Vector2{ 0.0f, 0.0f },
							  ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY))
		{

			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

			ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
			const auto wrapPosition = ImGui::GetContentRegionAvail().x - toggleSwitchWidth;
			ImGui::PushTextWrapPos(wrapPosition);

			const auto titleWidth =
				ImGui::CalcTextSize("Preconditioning Continuum Substraction", 0, true, wrapPosition).x;
			ImGui::TextWrapped("Preconditioning Continuum Substraction");
			ImGui::PopTextWrapPos();
			ImGui::PopFont();
			ImGui::SameLine(0.0f, wrapPosition - titleWidth);
			if (ui::ToggleSwitch(model_.params.contsub.enable, "##enable_preconditioning_continuum_substraction", "off",
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

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning continuum substraction order"))
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

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning continuum substraction padding"))
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

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning continuum substraction shift"))
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

			if (not isFilterEnabled_ or paramsFilter_.PassFilter("preconditioning continuum substraction threshhold"))
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

		ImGui::BeginDisabled(!model_.params.linker.enable);


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
			ui::HeadedTextOnly("Positivity");
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
	/*
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

		ImGui::BeginDisabled(!model_.params.reliability.enable);
		// TODO: content here
		ImGui::EndDisabled();
		ImGui::PopStyleColor();
		ImGui::PopStyleVar();
		ImGui::EndChild();
	}

	{
		ImGui::BeginChild("##mask_dilation", Vector2{ 0.0f, 0.0f },
						  ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY);
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

		ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
		const auto wrapPosition = ImGui::GetContentRegionAvail().x - toggleSwitchWidth;
		ImGui::PushTextWrapPos(wrapPosition);

		const auto titleWidth = ImGui::CalcTextSize("Mask Dilation", 0, true, wrapPosition).x;
		ImGui::TextWrapped("Mask Dilation");
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

		ImGui::BeginDisabled(!model_.params.dilation.enable);
		// TODO: content here
		ImGui::EndDisabled();
		ImGui::PopStyleColor();
		ImGui::PopStyleVar();
		ImGui::EndChild();
	}

	{
		ImGui::BeginChild("##parametrisation", Vector2{ 0.0f, 0.0f },
						  ImGuiChildFlags_FrameStyle | ImGuiChildFlags_AutoResizeY);
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.solidBackgroundFillColorBaseBrush);

		ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
		const auto wrapPosition = ImGui::GetContentRegionAvail().x - toggleSwitchWidth;
		ImGui::PushTextWrapPos(wrapPosition);

		const auto titleWidth = ImGui::CalcTextSize("Parametrisation", 0, true, wrapPosition).x;
		ImGui::TextWrapped("Parametrisation");
		ImGui::PopTextWrapPos();
		ImGui::PopFont();
		ImGui::SameLine(0.0f, wrapPosition - titleWidth);
		if (ui::ToggleSwitch(model_.params.parameter.enable, "##enable_parametrisation", "off", "on"))
		{
			model_.params.parameter.enable = !model_.params.parameter.enable;
		}

		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);
		ImGui::TextWrapped(
			"If enabled, SoFiA will attempt to convert relevant parameters to physical units. This involves "
			"conversion of channel widths to frequency/velocity units and division of flux-based parameters by "
			"the solid angle of the beam. For this to work, the relevant header parameters, including CTYPE3, "
			"CDELT3, BMAJ and BMIN, must have been correctly set. It is further assumed that the beam does not "
			"vary with frequency or position.");
		ImGui::PopStyleColor();

		ImGui::BeginDisabled(!model_.params.parameter.enable);
		// TODO: content here
		ImGui::EndDisabled();
		ImGui::PopStyleColor();
		ImGui::PopStyleVar();
		ImGui::EndChild();
	}
	*/
	ImGui::PopStyleVar();

	ImGui::EndDisabled();
}

auto SoFiaSearchView::drawFilterFormContentOld() -> void
{
	const auto disableInteraction = !model_.interactionEnabled;
	const auto hasProject = applicationContext_->selectedProject_.has_value();


	ImGui::BeginDisabled(disableInteraction);

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
					static_cast<int>(dimensions[2] - model_.selectedLocalRegion.upper.z * dimensions[2]) };

	model_.params.input.region.upper =
		owl::vec3i{ static_cast<int>(model_.selectedLocalRegion.upper.x * dimensions[0]),
					static_cast<int>(model_.selectedLocalRegion.upper.y * dimensions[1]),
					static_cast<int>(dimensions[2] - model_.selectedLocalRegion.lower.z * dimensions[2]) };

	model_.params.input.region.lower = owl::clamp(model_.params.input.region.lower, model_.params.input.region.upper);
	model_.params.input.region.upper =
		owl::clamp(model_.params.input.region.upper, model_.params.input.region.lower, dimensions);

	const auto btnDefaultSize = ImGui::CalcTextSize(ICON_LC_UNDO_2).x + ImGui::GetStyle().FramePadding.x * 2.0f;

	if (ImGui::CollapsingHeader("Input", ImGuiTreeNodeFlags_DefaultOpen))
	{

		ImGui::Checkbox("Show Subregion Tool", &model_.showRoiGizmo);
		ImGui::SameLine();
		if (ResetButtonOnSameLine(btnDefaultSize))
		{
			resetSelection();
		}

		ImGui::PushID("InputSettings");
		ImGui::Text("Region");

		ImGui::BeginDisabled(true);

		ImGui::DragInt3("Min", &model_.params.input.region.lower.x);


		ImGui::DragInt3("Max", &model_.params.input.region.upper.x);


		ImGui::EndDisabled();
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Preconditioning Continuum Substraction", ImGuiTreeNodeFlags_None))
	{

		ImGui::PushID("PreconditioningContinuumSettings");
		ImGui::Checkbox("Enabled##contsub", &model_.params.contsub.enable);
		ImGui::SameLine();
		HelpMarker("If enabled, SoFiA will try to subtract any residual continuum emission from the data cube prior to "
				   "source finding by fitting and subtracting a polynomial of order 0 (offset) or 1 (offset + slope). "
				   "The order of the polynomial is defined by contsub.order.");
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

		DragIntInputWidget(
			"Padding", &model_.params.contsub.padding, 1, 0, 1000 * 1000, 3, btnDefaultSize,
			"The amount of additional padding (in channels) applied to either side of channels excluded from the fit.");
		DragIntInputWidget("Shift", &model_.params.contsub.shift, 1, 1, 1000 * 1000, 4, btnDefaultSize,
						   "The number of channels by which the spectrum will be shifted (symmetrically in both "
						   "directions) before self-subtraction.");

		DragFloatInputWidget("Threshold", &model_.params.contsub.threshold, 0.1f, 0.0f, 1000.0f, 2.0f, btnDefaultSize,
							 "Relative clipping threshold. All channels with a flux density > contsub.threshold times "
							 "the noise will be clipped and excluded from the polynomial fit.");

		if (!model_.params.contsub.enable)
		{
			ImGui::EndDisabled();
		}

		ImGui::PopID();
	}

	ImGui::BeginDisabled(true);
	if (ImGui::CollapsingHeader("Preconditioning Flagging", ImGuiTreeNodeFlags_None))
	{
		ImGui::PushID("PreconditioningFlaggingSettings");
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

		ImGui::PopID();
	}
	ImGui::EndDisabled();


	if (ImGui::CollapsingHeader("Preconditioning Ripple Filter", ImGuiTreeNodeFlags_None))
	{

		ImGui::PushID("PreconditioningRippleSettings");

		ImGui::Checkbox("Enabled##Ripple", &model_.params.ripple.enable);
		ImGui::SameLine();
		HelpMarker("If set to true, then the ripple filter will be applied to the data cube prior to source finding. "
				   "The filter works by measuring and subtracting either the mean or median across a running window. "
				   "This can be useful if a DC offset or spatial/spectral ripple is present in the data.");
		if (!model_.params.ripple.enable)
		{
			ImGui::BeginDisabled(true);
		}

		DragIntInputWidget(
			"Grid XY", &model_.params.ripple.gridXY, 2, 0, 1001, 0, btnDefaultSize,
			"Spatial grid separation in pixels for the running window used in the ripple filter. The value must be an "
			"odd integer value and specifies the spatial step by which the window is moved. Alternatively, it can be "
			"set to 0, in which case it will default to half the spatial window size (see rippleFilter.windowXY).",
			true);

		DragIntInputWidget(
			"Grid Z", &model_.params.ripple.gridZ, 2, 0, 1001, 0, btnDefaultSize,
			"Spectral grid separation in channels for the running window used in the ripple filter. The value must be "
			"an odd integer value and specifies the spectral step by which the window is moved. Alternatively, it can "
			"be set to 0, in which case it will default to half the spectral window size (see rippleFilter.windowZ).",
			true);


		ImGui::Checkbox("Interpolate", &model_.params.ripple.interpolate);
		ImGui::SameLine();
		HelpMarker("Controls whether the mean or median should be measured and subtracted in the running window of the "
				   "ripple filter. The median is strongly recommended, as it is more robust.");
		if (ResetButtonOnSameLine(btnDefaultSize, &model_.params.ripple.interpolate, false))
		{
			model_.params.ripple.interpolate = false;
		}

		const char* rippleStatisticItems[] = { "median", "mean" };
		static auto rippleStatisticItem = 0;
		ImGui::Text("Statistic");
		ImGui::SameLine();
		HelpMarker("Controls whether the mean or median should be measured and subtracted in the running window of the "
				   "ripple filter. The median is strongly recommended, as it is more robust.");
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
		DragIntInputWidget("Window Z", &model_.params.ripple.windowZ, 2, 1, 100001, 1, btnDefaultSize,
						   "Spectral size in channels of the running window used in the ripple filter. The size must "
						   "be an odd integer number.",
						   true);

		if (!model_.params.ripple.enable)
		{
			ImGui::EndDisabled();
		}
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Preconditioning Noise Scaling", ImGuiTreeNodeFlags_None))
	{
		ImGui::PushID("NoiseScaleSettings");

		ImGui::Checkbox("Enabled##Noise", &model_.params.scaleNoise.enable);
		ImGui::SameLine();
		HelpMarker(
			"If set to true, noise scaling will be enabled. The purpose of the noise scaling modules is to measure the "
			"noise level in the input cube and then divide the input cube by the noise. This can be used to correct "
			"for spatial or spectral noise variations across the input cube prior to running the source finder.");

		if (!model_.params.scaleNoise.enable)
		{
			ImGui::BeginDisabled(true);
		}
		const char* fluxRangeItems[] = { "negative", "positive", "full" };
		static auto fluxRangeItem = 0;
		ComboWidget("Flux Range", fluxRangeItems, 3, &fluxRangeItem, &model_.params.scaleNoise.fluxRange, 0,
					btnDefaultSize,
					"Flux range to be used in the noise measurement. If set to negative or positive, only pixels with "
					"negative or positive flux will be used, respectively. This can be useful to prevent real emission "
					"or artefacts from affecting the noise measurement. If set to full, all pixels will be used in the "
					"noise measurement irrespective of their flux.");

		DragIntInputWidget("Grid XY", &model_.params.scaleNoise.gridXY, 2, 0, 1001, 0, btnDefaultSize,
						   "Size of the spatial grid across which noise measurement window will be moved across the "
						   "data cube. It must be an odd integer value. If set to 0 instead, the spatial grid size "
						   "will default to half the spatial window size.",
						   true);

		DragIntInputWidget("Grid Z", &model_.params.scaleNoise.gridZ, 2, 0, 1001, 0, btnDefaultSize,
						   "Size of the spectral grid across which noise measurement window will be moved across the "
						   "data cube. It must be an odd integer value. If set to 0 instead, the spectral grid size "
						   "will default to half the spectral window size.",
						   true);

		ImGui::Checkbox("Interpolate", &model_.params.scaleNoise.interpolate);
		ImGui::SameLine();
		HelpMarker(
			"If set to true, linear interpolation will be used to interpolate the measured local noise values in "
			"between grid points. If set to false, the entire grid cell will instead be filled with the measured noise "
			"value.");
		ResetButtonOnSameLine(btnDefaultSize, &model_.params.scaleNoise.interpolate, false);

		const char* scaleNoiseModeItems[] = { "spectral", "local" };
		static auto scaleNoiseModeItem = 0;
		ComboWidget("Mode", scaleNoiseModeItems, 2, &scaleNoiseModeItem, &model_.params.scaleNoise.mode, 0,
					btnDefaultSize,
					"Noise scaling mode. If set to spectral, the noise level will be determined for each spectral "
					"channel by measuring the noise within each image plane. This is useful for data cubes where the "
					"noise varies with frequency. If set to local, the noise level will be measured locally in window "
					"running across the entire cube in all three dimensions. This is useful for data cubes with more "
					"complex noise variations, such as interferometric images with primary-beam correction applied.");


		ImGui::Checkbox("ScFind", &model_.params.scaleNoise.scfind);
		ImGui::SameLine();
		HelpMarker("If true and global or local noise scaling is enabled, then noise scaling will additionally be "
				   "applied after each smoothing operation in the S+C finder. This might be useful in certain "
				   "situations where large-scale artefacts are present in interferometric data. However, this feature "
				   "should be used with great caution, as it has the potential to do more harm than good.");
		ResetButtonOnSameLine(btnDefaultSize, &model_.params.scaleNoise.scfind, false);

		const char* scaleNoiseStatisticItems[] = { "std", "mad", "gauss" };
		static auto scaleNoiseStatisticItem = 1;
		ComboWidget(
			"Statistic", scaleNoiseStatisticItems, 3, &scaleNoiseStatisticItem, &model_.params.scaleNoise.statistic, 1,
			btnDefaultSize,
			"Statistic to be used in the noise measurement process. Possible values are std, mad and gauss for "
			"standard deviation, median absolute deviation and Gaussian fitting to the flux histogram, respectively. "
			"Standard deviation is by far the fastest algorithm, but it is also the least robust one with respect to "
			"emission and artefacts in the data. Median absolute deviation and Gaussian fitting are far more robust in "
			"the presence of strong, extended emission or artefacts, but will usually take longer.");

		DragIntInputWidget("Window XY", &model_.params.scaleNoise.windowXY, 2, 0, 1001, 25, btnDefaultSize,
						   "Spatial size of the window used in determining the local noise level. It must be an odd "
						   "integer value. If set to 0, the pipeline will use the default value instead.",
						   true);


		DragIntInputWidget("Window Z", &model_.params.scaleNoise.windowZ, 2, 0, 1001, 15, btnDefaultSize,
						   "Spectral size of the window used in determining the local noise level.It must be an odd "
						   "integer value.If set to 0, the pipeline will use the default value instead.",
						   true);

		if (!model_.params.scaleNoise.enable)
		{
			ImGui::EndDisabled();
		}
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Source Finding", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::PushID("SourceFindingSettings");

		ImGui::Checkbox("Enabled##ScFind", &model_.params.scfind.enable);
		ImGui::SameLine();
		HelpMarker("If set to true, the Smooth + Clip (S+C) finder will be enabled. The S+C finder operates by "
				   "iteratively smoothing the data cube with a user-defined set of smoothing kernels, measuring the "
				   "noise level on each smoothing scale, and adding all pixels with an absolute flux above a "
				   "user-defined relative threshold to the source detection mask.");

		if (!model_.params.scfind.enable)
		{
			ImGui::BeginDisabled(true);
		}

		const char* fluxRangeItems[] = { "negative", "positive", "full" };
		static auto fluxRangeItem = 0;
		ComboWidget("Flux Range", fluxRangeItems, 3, &fluxRangeItem, &model_.params.scfind.fluxRange, 0, btnDefaultSize,
					"Flux range to be used in the noise measurement. If set to negative or positive, only pixels with "
					"negative or positive flux will be used, respectively. This can be useful to prevent real emission "
					"or artefacts from affecting the noise measurement. If set to full, all pixels will be used in the "
					"noise measurement irrespective of their flux.");

		/*
		 * scfind.kernelsXY
		 * scfind.kernelsZ
		 */
		ImGui::PushID("ScFindKernelXY");

		ImGui::Text("Kernels XY");
		ImGui::SameLine();
		HelpMarker(
			"Comma-separated list of spatial Gaussian kernel sizes to apply. The individual kernel sizes must be "
			"floating-point values and denote the full width at half maximum (FWHM) of the Gaussian used to smooth the "
			"data in the spatial domain. A value of 0 means that no spatial smoothing will be applied.");
		ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - btnDefaultSize);
		if (ui::Button(ICON_LC_UNDO_2))
		{
			model_.params.scfind.kernelsXY = { 0, 3, 6 };
		}
		if (ImGui::BeginChild("##KernelsXY",
							  ImVec2(-FLT_MIN, ImGui::GetFontSize() * (model_.params.scfind.kernelsXY.size() + 2)),
							  ImGuiChildFlags_Border))
		{
			for (auto i = 0; i < model_.params.scfind.kernelsXY.size(); ++i)
			{
				auto& kernel = model_.params.scfind.kernelsXY[i];
				ImGui::PushID(i);
				ImGui::Text(std::format("{}", kernel).c_str());
				ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - btnDefaultSize);
				if (ui::Button(ICON_LC_X))
				{
					model_.params.scfind.kernelsXY.erase(model_.params.scfind.kernelsXY.begin() + i);
				}
				ImGui::PopID();
			}
		}
		ImGui::EndChild();

		static int inputXY = 7;
		ImGui::InputInt("", &inputXY, 1, 5);
		inputXY = std::max(0, inputXY);
		ImGui::SameLine();
		if (ui::Button("Add##AddKernelXY"))
		{
			if (model_.params.scfind.kernelsXY.end() == std::ranges::find(model_.params.scfind.kernelsXY, inputXY))
			{
				model_.params.scfind.kernelsXY.push_back(inputXY++);
			}
		}

		ImGui::PopID();

		ImGui::PushID("ScFindKernelZ");
		ImGui::Text("Kernels Z");
		ImGui::SameLine();
		HelpMarker("Comma-separated list of spectral Boxcar kernel sizes to apply. The individual kernel sizes must be "
				   "odd integer values of 3 or greater and denote the full width of the Boxcar filter used to smooth "
				   "the data in the spectral domain. A value of 0 means that no spectral smoothing will be applied.");
		ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - btnDefaultSize);
		if (ui::Button(ICON_LC_UNDO_2))
		{
			model_.params.scfind.kernelsZ = { 0, 3, 7, 15 };
		}
		if (ImGui::BeginChild("##KernelsZ",
							  ImVec2(-FLT_MIN, ImGui::GetFontSize() * (model_.params.scfind.kernelsZ.size() + 2)),
							  ImGuiChildFlags_Border))
		{
			for (auto i = 0; i < model_.params.scfind.kernelsZ.size(); ++i)
			{
				auto& kernel = model_.params.scfind.kernelsZ[i];
				ImGui::PushID(i);
				ImGui::Text(std::format("{}", kernel).c_str());
				ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - btnDefaultSize);
				if (ui::Button(ICON_LC_X))
				{
					model_.params.scfind.kernelsZ.erase(model_.params.scfind.kernelsZ.begin() + i);
				}
				ImGui::PopID();
			}
		}
		ImGui::EndChild();

		static int inputZ = 17;
		inputZ = std::max(0, inputZ);

		inputZ = (inputZ > 0 && inputZ % 2 == 0) ? inputZ + 1 : inputZ;
		ImGui::InputInt("", &inputZ, 2, 5);

		ImGui::SameLine();
		if (ui::Button("Add##AddKernelZ"))
		{
			if (model_.params.scfind.kernelsZ.end() == std::ranges::find(model_.params.scfind.kernelsZ, inputZ))
			{
				model_.params.scfind.kernelsZ.push_back(inputZ++);
			}
		}

		ImGui::PopID();


		DragFloatInputWidget(
			"Replacement", &model_.params.scfind.replacement, 0.01f, -1.0f, 9999.0f, 2.0f, btnDefaultSize,
			"Before smoothing the data cube during an S+C iteration, every pixel in the data cube that was already "
			"detected in a previous iteration will be replaced by this value multiplied by the original noise level in "
			"the non-smoothed data cube, while keeping the original sign of the data value. This feature can be "
			"disabled altogether by specifying a value of < 0.");


		DragFloatInputWidget("Threshold", &model_.params.scfind.threshold, 0.01f, 0.0f, 100.0f, 5.0f, btnDefaultSize,
							 "Flux threshold to be used by the S+C finder relative to the measured noise level in each "
							 "smoothing iteration. In practice, values in the range of about 3 to 5 have proven to be "
							 "useful in most situations, with lower values in that range requiring use of the "
							 "reliability filter to reduce the number of false detections.");


		const char* scaleNoiseStatisticItems[] = { "std", "mad", "gauss" };
		static auto scaleNoiseStatisticItem = 1;
		ComboWidget(
			"Statistic", scaleNoiseStatisticItems, 3, &scaleNoiseStatisticItem, &model_.params.scfind.statistic, 1,
			btnDefaultSize,
			"Statistic to be used in the noise measurement process. Possible values are std, mad and gauss for "
			"standard deviation, median absolute deviation and Gaussian fitting to the flux histogram, respectively. "
			"Standard deviation is by far the fastest algorithm, but it is also the least robust one with respect to "
			"emission and artefacts in the data. Median absolute deviation and Gaussian fitting are far more robust in "
			"the presence of strong, extended emission or artefacts, but will usually take longer.");


		if (!model_.params.scfind.enable)
		{
			ImGui::EndDisabled();
		}

		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Linking", ImGuiTreeNodeFlags_None))
	{
		ImGui::PushID("LinkingSettings");

		ImGui::Checkbox("Enabled##Linking", &model_.params.linker.enable);
		ImGui::SameLine();
		HelpMarker("If true, then the linker will be run to merge the pixels detected by the source finder into "
				   "coherent detections that can then be parameterised and catalogued. If false, the pipeline will be "
				   "terminated after source finding, and no catalogue or source products will be created. Disabling "
				   "the linker can be useful if only the raw mask from the source finder is needed.");

		if (!model_.params.linker.enable)
		{
			ImGui::BeginDisabled(true);
		}

		ImGui::Checkbox("Keep Negative##KeepNeg", &model_.params.linker.keepNegative);
		ImGui::SameLine();
		HelpMarker("If set to true, then the linker will not discard detections with negative flux. Reliability "
				   "filtering must be disabled for negative sources to be retained. Also note that negative sources "
				   "will not appear in moment 1 and 2 maps. This option should only ever be used for testing or "
				   "debugging purposes, but never in production mode.");


		DragFloatInputWidget(
			"Max Fill", &model_.params.linker.maxFill, 0.01f, 0.0f, 100.0f, .0f, btnDefaultSize,
			"Maximum allowed filling factor of a source within its rectangular bounding box, defined as the number of "
			"spatial and spectral pixels that make up the source divided by the number of pixels in the bounding box. "
			"The default value of 0.0 disables maximum filling factor filtering.");

		DragIntInputWidget("Max Pixels", &model_.params.linker.maxPixels, 1, 0, 999999, 0, btnDefaultSize,
						   "Maximum allowed number of spatial and spectral pixels that a source must not exceed.The "
						   "default value of 0 disables maximum size filtering.",
						   false);

		DragIntInputWidget(
			"Max Size XY", &model_.params.linker.maxSizeXY, 1, 0, 999999, 0, btnDefaultSize,
			"Maximum size of sources in the spatial dimension in pixels.Sources that exceed this limit will be "
			"discarded by the linker.If the value is set to 0, maximum size filtering will be disabled.",
			false);

		DragIntInputWidget(
			"Max Size Z", &model_.params.linker.maxSizeZ, 1, 0, 999999, 0, btnDefaultSize,
			"Maximum size of sources in the spectral dimension in pixels. Sources that exceed this limit will be "
			"discarded by the linker. If the value is set to 0, maximum size filtering will be disabled.",
			false);


		DragFloatInputWidget(
			"Min Fill", &model_.params.linker.minFill, 0.01f, 0.0f, 100.0f, .0f, btnDefaultSize,
			"Minimum allowed filling factor of a source within its rectangular bounding box, defined as the number of "
			"spatial and spectral pixels that make up the source divided by the number of pixels in the bounding box. "
			"The default value of 0.0 disables minimum filling factor filtering.");

		DragIntInputWidget("Min Pixels", &model_.params.linker.minPixels, 1, 0, 999999, 0, btnDefaultSize,
						   "Minimum allowed number of spatial and spectral pixels that a source must have. The default "
						   "value of 0 disables minimum size filtering.",
						   false);

		DragIntInputWidget("Min Size XY", &model_.params.linker.minSizeXY, 1, 1, 999999, 5, btnDefaultSize,
						   "Minimum size of sources in the spatial dimension in pixels. Sources that fall below this "
						   "limit will be discarded by the linker.",
						   false);

		DragIntInputWidget("Min Size Z", &model_.params.linker.minSizeZ, 1, 1, 999999, 5, btnDefaultSize,
						   "Minimum size of sources in the spectral dimension in pixels. Sources that fall below this "
						   "limit will be discarded by the linker.",
						   false);

		ImGui::Checkbox("Positivity##positivity", &model_.params.linker.positivity);
		ImGui::SameLine();
		HelpMarker("If set to true, then the linker will only merge positive pixels and discard all negative pixels by "
				   "removing them from the mask. This option should be used with extreme caution and will render the "
				   "reliability filter useless. It can be useful, though, if there are significant negative artefacts "
				   "such as residual sidelobes in the data.");

		DragIntInputWidget("Radius XY", &model_.params.linker.radiusXY, 1, 1, 999999, 1, btnDefaultSize,
						   "Maximum merging length in the spatial dimension. Pixels with a separation of up to this "
						   "value will be merged into the same source.",
						   false);

		DragIntInputWidget("Radius Z", &model_.params.linker.radiusZ, 1, 1, 999999, 1, btnDefaultSize,
						   "Maximum merging length in the spectral dimension. Pixels with a separation of up to this "
						   "value will be merged into the same source.",
						   false);

		if (!model_.params.linker.enable)
		{
			ImGui::EndDisabled();
		}

		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Reliability", ImGuiTreeNodeFlags_None))
	{
		ImGui::PushID("ReliabilitySettings");

		ImGui::Checkbox("Enabled##Reliability", &model_.params.reliability.enable);
		ImGui::SameLine();
		HelpMarker("If set to true, reliability calculation and filtering will be enabled. This will determine the "
				   "reliability of each detection with positive total flux by comparing the density of positive and "
				   "negative detections in a three-dimensional parameter space. Sources below the specified "
				   "reliability threshold will then be discarded. Note that this will require a sufficient number of "
				   "negative detections, which can usually be achieved by setting the source finding threshold to "
				   "somewhere around 3 to 4 times the noise level.");

		if (!model_.params.reliability.enable)
		{
			ImGui::BeginDisabled(true);
		}

		ImGui::Checkbox("Auto Kernel##autoKernel", &model_.params.reliability.autoKernel);
		ImGui::SameLine();
		HelpMarker(
			"If set to true, SoFiA will try to automatically determine the optimal reliability kernel scale factor by "
			"iteratively increasing the kernel size until the absolute value of the median of the Skellam distribution "
			"decreases below reliability.tolerance. If the algorithm fails to converge after reliability.iterations "
			"steps, then the default value of reliability.scaleKernel will be used instead.");

		// SKipped
		// reliability.catalog
		// reliability.debug

		DragIntInputWidget(
			"Iterations", &model_.params.reliability.iterations, 1, 1, 999, 30, btnDefaultSize,
			"Maximum number of iterations for the reliability kernel auto-scaling algorithm to converge. If "
			"convergence is not achieved, then reliability.scaleKernel will instead be applied.",
			false);

		DragIntInputWidget(
			"Min Pixels", &model_.params.reliability.minPixels, 1, 0, 999999, 1, btnDefaultSize,
			"Minimum total number of spatial and spectral pixels within the source mask for detections to be "
			"considered reliable. The reliability of any detection with fewer pixels will be set to zero by default.",
			false);

		DragFloatInputWidget(
			"Min SNR", &model_.params.reliability.minSNR, 0.01f, 0.0f, 100.0f, 3.0f, btnDefaultSize,
			"Lower signal-to-noise limit for reliable sources. Detections that fall below this threshold will be "
			"deemed unreliable and assigned a reliability of 0. The value denotes the integrated signal-to-noise "
			"ratio, SNR = F_sum / [RMS * sqrt(N * Ω)], of the source, where Ω is the solid angle (in pixels) of the "
			"point spread function of the data, N is the number of spatial and spectral pixels of the source, F_sum is "
			"the summed flux density and RMS is the local RMS noise level (assumed to be constant). Note that the "
			"spectral resolution is assumed to be equal to the channel width.");

		const char* reliabilityParamItems[9] = { "peak", "sum", "mean", "chan", "pix", "fill", "std", "skew", "kurt" };
		static std::array<bool, 9> reliabilityParamItemBoxes = { true,	true,  true,  false, false,
																 false, false, false, false };
		ImGui::Text("Parameters");
		ImGui::SameLine();
		HelpMarker("Parameter space to be used in deriving the reliability of detections. This must be a list of "
				   "parameters the number of which defines the dimensionality of the parameter space. Possible values "
				   "are peak for the peak flux density, sum for the summed flux density, mean for mean flux density, "
				   "chan for the number of spectral channels, pix for the total number of spatial and spectral pixels, "
				   "fill for the filling factor, std for the standard deviation, skew for the skewness and kurt for "
				   "the kurtosis across the source mask. Flux densities will be divided by the global RMS noise level. "
				   "peak, sum, mean, pix and fill will be logarithmic, all other parameters linear.");
		ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - btnDefaultSize);
		if (ui::Button(ICON_LC_UNDO_2))
		{
			reliabilityParamItemBoxes = { true, true, true, false, false, false, false, false, false };
		}
		if (ImGui::BeginChild("##Parameters", ImVec2(-FLT_MIN, ImGui::GetFontSize() * 5), ImGuiChildFlags_Border))
		{
			ImGuiMultiSelectIO* ms_io =
				ImGui::BeginMultiSelect(ImGuiMultiSelectFlags_NoAutoSelect | ImGuiMultiSelectFlags_NoAutoClear, -1,
										static_cast<int>(reliabilityParamItemBoxes.size()));
			ImGuiSelectionExternalStorage storage_wrapper;
			storage_wrapper.UserData = reliabilityParamItemBoxes.data();
			storage_wrapper.AdapterSetItemSelected = [](ImGuiSelectionExternalStorage* self, int n, bool selected)
			{
				bool* array = (bool*)self->UserData;
				array[n] = selected;
			};
			storage_wrapper.ApplyRequests(ms_io);
			for (int n = 0; n < 9; n++)
			{
				ImGui::SetNextItemSelectionUserData(n);
				ImGui::Checkbox(reliabilityParamItems[n], &reliabilityParamItemBoxes[n]);
			}
			ms_io = ImGui::EndMultiSelect();
			storage_wrapper.ApplyRequests(ms_io);
		}

		ImGui::EndChild();

		model_.params.reliability.parameters.clear();
		for (int i = 0; i < reliabilityParamItemBoxes.size(); ++i)
		{
			if (reliabilityParamItemBoxes[i])
			{
				model_.params.reliability.parameters.push_back(reliabilityParamItems[i]);
			}
		}

		// Skipped
		// reliability.plot

		DragFloatInputWidget("Scale Kernel", &model_.params.reliability.scaleKernel, 0.001f, -100.0f, 100.0f, 0.4f,
							 btnDefaultSize,
							 "When estimating the density of positive and negative detections in parameter space, the "
							 "size of the Gaussian kernel used in this process is determined from the covariance of "
							 "the distribution of negative detections in parameter space. This parameter setting can "
							 "be used to scale that kernel by a constant factor.");

		DragFloatInputWidget("Threshold", &model_.params.reliability.threshold, 0.001f, 0.0f, 1.0f, 0.9f,
							 btnDefaultSize,
							 "Reliability threshold in the range of 0 to 1. Sources with a reliability below this "
							 "threshold will be discarded.");

		DragFloatInputWidget(
			"Tolerance ", &model_.params.reliability.tolerance, 0.001f, -100.0f, 100.0f, 0.05f, btnDefaultSize,
			"Convergence tolerance for the reliability kernel auto-scaling algorithm. Convergence is achieved when the "
			"absolute value of the median of the Skellam distribution drops below this tolerance.",
			"%.4f");


		if (!model_.params.reliability.enable)
		{
			ImGui::EndDisabled();
		}


		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Mask Dilation", ImGuiTreeNodeFlags_None))
	{
		ImGui::PushID("MaskDilationSettings");
		ImGui::Checkbox("Enabled##MaskDilation", &model_.params.dilation.enable);
		ImGui::SameLine();
		HelpMarker("Set to true to enable source mask dilation whereby the mask of each source will be grown outwards "
				   "until the resulting increase in integrated flux drops below a given threshold or the maximum "
				   "number of iterations is reached.");

		if (!model_.params.dilation.enable)
		{
			ImGui::BeginDisabled(true);
		}

		DragIntInputWidget("Iterations XY", &model_.params.dilation.iterationsXY, 1, 1, 999999, 10, btnDefaultSize,
						   "Sets the maximum number of spatial iterations for the mask dilation algorithm. Once this "
						   "number of iterations has been reached, mask dilation in the spatial plane will stop even "
						   "if the flux increase still exceeds the threshold set by dilation.threshold.",
						   false);

		DragIntInputWidget("Iterations Z", &model_.params.dilation.iterationsZ, 1, 1, 999999, 5, btnDefaultSize,
						   "Sets the maximum number of spectral iterations for the mask dilation algorithm. Once this "
						   "number of iterations has been reached, mask dilation along the spectral axis will stop "
						   "even if the flux increase still exceeds the threshold set by dilation.threshold.",
						   false);

		DragFloatInputWidget(
			"Threshold", &model_.params.dilation.threshold, 0.0001f, -5.0f, 5.0f, 0.001f, btnDefaultSize,
			"If a positive value is provided, mask dilation will end when the increment in the integrated flux during "
			"a single iteration drops below this value times the total integrated flux (from the previous iteration), "
			"or when the maximum number of iterations has been reached. Specifying a negative threshold will disable "
			"flux checking altogether and always carry out the maximum number of iterations.",
			"%.5f");


		if (!model_.params.dilation.enable)
		{
			ImGui::EndDisabled();
		}


		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Parametrisation", ImGuiTreeNodeFlags_None))
	{
		ImGui::PushID("ParametrisationSettings");
		ImGui::Checkbox("Enabled##Parametrisation", &model_.params.parameter.enable);
		ImGui::SameLine();
		HelpMarker("If set to true, the parametrisation module will be enabled to measure the basic parameters of each "
				   "detected source.");

		if (!model_.params.parameter.enable)
		{
			ImGui::BeginDisabled(true);
		}

		// Skipped
		//  parameter.offset

		ImGui::Checkbox("Parametrisation##physical", &model_.params.parameter.physical);
		ImGui::SameLine();
		HelpMarker("If set to true, SoFiA will attempt to convert relevant parameters to physical units. This involves "
				   "conversion of channel widths to frequency/velocity units and division of flux-based parameters by "
				   "the solid angle of the beam. For this to work, the relevant header parameters, including CTYPE3, "
				   "CDELT3, BMAJ and BMIN, must have been correctly set. It is further assumed that the beam does not "
				   "vary with frequency or position.");

		ImGui::BeginDisabled(true);
		ImGui::Checkbox("Parametrisation##wcs", &model_.params.parameter.wcs);
		ImGui::SameLine();
		HelpMarker("If set to true, SoFiA will attempt to convert the source centroid position (x, y, z) to world "
				   "coordinates using the WCS information stored in the header. In addition, spectra and moment map "
				   "units will be converted from channels to WCS units as well.");
		ImGui::EndDisabled();

		if (!model_.params.parameter.enable)
		{
			ImGui::EndDisabled();
		}

		ImGui::PopID();
	}

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
