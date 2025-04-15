#include "Style.h"


#include "framework/ApplicationContext.h"
#include "imgui_stdlib.h"


enum class ItemState
{
	base,
	hovered,
	pressed,
	disabled
};


auto ui::Button(const char* label, const Vector2& size) -> bool
{
	ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2.0f);

	const auto& brush = ApplicationContext::getStyleBrush();
	const auto itemId = ImGui::GetID(label);
	const auto isHovered =
		ImGui::GetHoveredID() == itemId or ImGui::GetCurrentContext()->HoveredIdPreviousFrame == itemId;
	const auto isActive = ImGui::GetActiveID() == itemId;
	const auto isDisabled = ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled;

	auto result = false;
	auto state = ItemState::base;
	if (isHovered)
	{
		state = ItemState::hovered;
	}
	if (isActive)
	{
		state = ItemState::pressed;
	}
	if (isDisabled)
	{
		state = ItemState::disabled;
	}

	switch (state)
	{
	default:
	case ItemState::base:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(4);
		break;
	case ItemState::hovered:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(5);
		break;
	case ItemState::pressed:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrokeColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(5);
		break;
	case ItemState::disabled:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrokeColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(3);
		break;
	}

	ImGui::PopStyleVar();

	return result;
}

auto ui::AccentButton(const char* label, const Vector2& size) -> bool
{
	ImGui::SetNextItemAllowOverlap();
	const auto& brush = ApplicationContext::getStyleBrush();
	const auto itemId = ImGui::GetID(label);
	const auto isHovered = ImGui::GetHoveredID() == itemId;
	const auto isActive = ImGui::GetActiveID() == itemId;
	const auto isDisabled = ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled;

	auto result = false;
	auto state = ItemState::base;
	if (isHovered)
	{
		state = ItemState::hovered;
	}
	if (isActive)
	{
		state = ItemState::pressed;
	}
	if (isDisabled)
	{
		state = ItemState::disabled;
	}
	// state = ItemState::hovered;
	switch (state)
	{
	default:
	case ItemState::base:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.accentFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.accentControlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textOnAccentFillColorPrimaryBrush);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(3);
		break;
	case ItemState::hovered:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.accentFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.accentFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.accentFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.accentControlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textOnAccentFillColorPrimaryBrush);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(5);
		break;
	case ItemState::pressed:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.accentFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.accentFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlFillColorTransparentBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textOnAccentFillColorSecondaryBrush);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(4);
		break;
	case ItemState::disabled:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.accentFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlFillColorTransparentBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textOnAccentFillColorDisabledBrush);


		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(3);
		break;
	}
	return result;
}

auto ui::ToggleButton(const bool toogled, const char* label, const Vector2& size) -> bool
{
	ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2.0f);

	const auto& brush = ApplicationContext::getStyleBrush();
	const auto itemId = ImGui::GetID(label);
	const auto isHovered =
		ImGui::GetHoveredID() == itemId or ImGui::GetCurrentContext()->HoveredIdPreviousFrame == itemId;
	const auto isActive = ImGui::GetActiveID() == itemId;
	const auto isDisabled = ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled;

	auto result = false;
	auto state = ItemState::base;
	if (isHovered)
	{
		state = ItemState::hovered;
	}
	if (isActive or toogled)
	{
		state = ItemState::pressed;
	}
	if (isDisabled)
	{
		state = ItemState::disabled;
	}

	switch (state)
	{
	default:
	case ItemState::base:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(4);
		break;
	case ItemState::hovered:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(5);
		break;
	case ItemState::pressed:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.textControlElevationBorderFocusedBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(5);
		break;
	case ItemState::disabled:
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrokeColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(3);
		break;
	}

	ImGui::PopStyleVar();

	return result;
}

auto ui::ToggleSwitch(const bool isOn, const char* label, const char* option1 = "", const char* option2 = "") -> bool
{
	const auto isDisabled = ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled;

	const auto& brush = ApplicationContext::getStyleBrush();

	const auto types = std::array{ option1, option2 };

	ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 14.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 12.0f);

	if (isDisabled)
	{
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrongStrokeColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, brush.controlStrongStrokeColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, brush.controlStrongStrokeColorDisabledBrush);
	}
	else
	{
		if (isOn)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.accentFillColorDefaultBrush);
			ImGui::PushStyleColor(ImGuiCol_Border, brush.accentFillColorDefaultBrush);
			ImGui::PushStyleColor(ImGuiCol_SliderGrab, brush.controlFillColorDefaultBrush);
		}
		else
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDefaultBrush);
			ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrongStrokeColorDefaultBrush);
			ImGui::PushStyleColor(ImGuiCol_SliderGrab, brush.controlStrongStrokeColorDefaultBrush);
		}
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.accentFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.accentFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, brush.textFillColorSecondaryBrush);
	}
	static auto value = isOn ? 1 : 0;
	const auto isEdited = ImGui::SliderInt(label, &value, 0, 1, types[value], ImGuiSliderFlags_NoInput);
	const auto result = ImGui::IsItemFocused() ? isEdited : ImGui::IsItemClicked(ImGuiMouseButton_Left);

	ImGui::PopStyleColor(6);
	ImGui::PopStyleVar(3);
	return result;
}

auto ui::Selectable(const char* label, bool selected, ImGuiSelectableFlags flags, const Vector2& size) -> bool
{
	auto result = false;
	result = ImGui::Selectable(label, selected, flags, size);
	return result;
}

auto ui::InputText(const char* label, std::string* text, ImGuiInputTextFlags flags) -> bool
{
	const auto& brush = ApplicationContext::getStyleBrush();
	const auto itemId = ImGui::GetID(label);
	/*const auto isHovered =
		ImGui::GetHoveredID() == itemId or ImGui::GetCurrentContext()->HoveredIdPreviousFrame == itemId;*/
	const auto isActive = ImGui::GetActiveID() == itemId;
	const auto isDisabled = ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled;

	const auto borderSize = 2.0f;
	ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, borderSize);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
	ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrokeColorSecondaryBrush);
	ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, brush.accentFillColorSelectedTextBackgroundBrush);

	if (isDisabled)
	{
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.solidBackgroundFillColorBaseBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorSecondaryBrush);
	}

	const auto position = Vector2{ ImGui::GetCursorScreenPos() };
	ImGui::PushItemWidth(-1);
	const auto isEdited = ImGui::InputText(label, text, flags);
	ImGui::PopItemWidth();
	const auto inputBoxSize = Vector2{ ImGui::GetItemRectSize() };

	auto& drawList = *ImGui::GetForegroundDrawList();
	drawList.AddLine(
		position + Vector2{ 0.0f, inputBoxSize.y - borderSize }, position + inputBoxSize - Vector2{ 0.0f, borderSize },
		isActive ? brush.accentTextFillColorTertiaryBrush : brush.controlStrongStrokeColorDefaultBrush, borderSize);

	ImGui::PopStyleColor(5);
	ImGui::PopStyleVar(2);
	return isEdited;
}

auto ui::HeadedInputText(const std::string& header, const char* label, std::string* text, ImGuiInputTextFlags flags)
	-> bool
{
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vector2{ 12.0f, 8.0f });
	ImGui::Text(header.c_str());
	const auto result = ui::InputText(label, text, flags);
	ImGui::PopStyleVar();
	return result;
}

auto createDarkThemeBrush(const AccentColors& accentColors) -> StyleBrush
{
	auto brush = StyleBrush{};

	brush.textFillColorPrimaryBrush = Color{ 1.0f, 1.0f, 1.0f, 1.0f };
	brush.textFillColorSecondaryBrush = Color{ 0.8f, 0.8f, 0.8f, 1.0f };
	brush.textFillColorTertiaryBrush = Color{ 0.59f, 0.59f, 0.59f, 1.0f };
	brush.textFillColorDisabledBrush = Color{ 0.44f, 0.44f, 0.44f, 1.0f };
	brush.accentTextFillColorPrimaryBrush = accentColors.accentLight3;
	brush.accentTextFillColorSecondaryBrush = accentColors.accentLight3;
	brush.accentTextFillColorTertiaryBrush = accentColors.accentLight2;
	brush.accentTextFillColorDisabledBrush = Color{ 0.44f, 0.44f, 0.44f, 1.0f };
	brush.textOnAccentFillColorPrimaryBrush = Color{ 0.0f, 0.0f, 0.0f, 1.0f };
	brush.textOnAccentFillColorSecondaryBrush = Color{ 0.06f, 0.06f, 0.06f, 1.0f };
	brush.textOnAccentFillColorDisabledBrush = Color{ 0.59f, 0.59f, 0.59f, 1.0f };
	brush.textOnAccentFillColorSelectedTextBrush = Color{ 1.0f, 1.0f, 1.0f, 1.0f };

	brush.accentFillColorSelectedTextBackgroundBrush = accentColors.accentDark2;

	brush.solidBackgroundFillColorBaseBrush = Color{ 0.13f, 0.13f, 0.13f, 1.0f };
	brush.solidBackgroundFillColorBaseAltBrush = Color{ 0.04f, 0.04f, 0.04f, 1.0f };
	brush.solidBackgroundFillColorSecondaryBrush = Color{ 0.11f, 0.11f, 0.11f, 1.0f };
	brush.solidBackgroundFillColorTertiaryBrush = Color{ 0.16f, 0.16f, 0.16f, 1.0f };
	brush.solidBackgroundFillColorQuarternaryBrush = Color{ 0.17f, 0.17f, 0.17f, 1.0f };

	brush.accentAcrylicBackgroundFillColorBaseBrush = accentColors.accentDark2;
	brush.accentAcrylicBackgroundFillColorDefaultBrush = accentColors.accentDark1;

	brush.controlFillColorDefaultBrush = Color{ 0.18f, 0.18f, 0.18f, 1.0f };
	brush.controlFillColorSecondaryBrush = Color{ 0.2f, 0.2f, 0.2f, 1.0f };
	brush.controlFillColorTertiaryBrush = Color{ 0.15f, 0.15f, 0.15f, 1.0f };
	brush.controlFillColorDisabledBrush = Color{ 0.16f, 0.16f, 0.16f, 1.0f };
	brush.controlFillColorTransparentBrush = Color{ 0.13f, 0.13f, 0.13f, 1.0f };
	brush.controlFillColorInputActiveBrush = Color{ 0.12f, 0.12f, 0.12f, 1.0f };

	brush.controlStrokeColorDefaultBrush = Color{ 0.19f, 0.19f, 0.19f, 1.0f };
	brush.controlStrokeColorSecondaryBrush = Color{ 0.21f, 0.21f, 0.21f, 1.0f };
	brush.controlStrokeColorOnAccentDefaultBrush = Color{ 0.19f, 0.19f, 0.19f, 1.0f };
	brush.controlStrokeColorOnAccentSecondaryBrush = Color{ 0.11f, 0.11f, 0.11f, 1.0f };

	brush.controlElevationBorderBrush = Color{ 0.19f, 0.19f, 0.19f, 1.0f };
	brush.textControlElevationBorderFocusedBrush = accentColors.accent;
	brush.accentControlElevationBorderBrush = brush.controlElevationBorderBrush;

	brush.cardStrokeColorDefaultBrush = Color{ 0.11f, 0.11f, 0.11f, 1.0f };
	brush.cardStrokeColorDefaultSolidBrush = Color{ 0.11f, 0.11f, 0.11f, 1.0f };

	brush.controlStrongStrokeColorDisabledBrush = Color{ 0.26f, 0.26f, 0.26f, 1.0f };
	brush.controlStrongStrokeColorDefaultBrush = Color{ 0.6f, 0.6f, 0.6f, 1.0f };


	brush.accentFillColorDefaultBrush = accentColors.accentDark1;
	brush.accentFillColorSecondaryBrush = accentColors.accentDark1;
	brush.accentFillColorSecondaryBrush.a = static_cast<uint32_t>(0.9f * 255);
	brush.accentFillColorTertiaryBrush = accentColors.accentDark1;
	brush.accentFillColorTertiaryBrush.a = static_cast<uint32_t>(0.8f * 255);
	brush.accentFillColorDisabledBrush = Color{ 0.44f, 0.44f, 0.44f, 1.0f };

	brush.cardBackgroundFillColorDefaultBrush = Color{ 0.17f, 0.17f, 0.17f, 1.0f };
	brush.layerFillColorDefaultBrush = Color{ 0.15f, 0.15f, 0.15f, 1.0f };

	return brush;
}
