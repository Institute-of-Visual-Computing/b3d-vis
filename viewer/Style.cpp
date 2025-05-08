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

auto ui::SignalButton(const SignalState signal, const char* label, const Vector2& size) -> bool
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

	auto backgroundColor = Color{};
	auto forgroundColor = Color{};

	switch (signal)
	{
	case SignalState::success:
		backgroundColor = brush.systemFillColorSuccessBackgroundBrush;
		forgroundColor = brush.systemFillColorSuccessBrush;
		break;
	case SignalState::caution:
		backgroundColor = brush.systemFillColorCautionBackgroundBrush;
		forgroundColor = brush.systemFillColorCautionBrush;
		break;
	case SignalState::critical:
		backgroundColor = brush.systemFillColorCriticalBackgroundBrush;
		forgroundColor = brush.systemFillColorCriticalBrush;
		break;
	}

	switch (state)
	{
	default:
	case ItemState::base:
		ImGui::PushStyleColor(ImGuiCol_Button, backgroundColor);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, forgroundColor);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(4);
		break;
	case ItemState::hovered:
		ImGui::PushStyleColor(ImGuiCol_Button, backgroundColor);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, forgroundColor);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(5);
		break;
	case ItemState::pressed:
		ImGui::PushStyleColor(ImGuiCol_Button, backgroundColor);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrokeColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, forgroundColor);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(5);
		break;
	case ItemState::disabled:
		ImGui::PushStyleColor(ImGuiCol_Button, backgroundColor);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrokeColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, forgroundColor);

		result = ImGui::Button(label, size);

		ImGui::PopStyleColor(3);
		break;
	}

	ImGui::PopStyleVar();

	return result;
}

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
	constexpr auto borderSize = 2.0f;
	constexpr auto roundingRadius = 14.0f;
	ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, borderSize);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, roundingRadius);
	ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 12.0f);

	if (isDisabled)
	{
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrongStrokeColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, brush.controlStrongStrokeColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, brush.controlStrongStrokeColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorDisabledBrush);
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
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);
	}
	auto value = isOn ? 1 : 0;
	ImGui::SetNextItemWidth(roundingRadius * 4.0f + borderSize * 2.0f);
	const auto isEdited = ImGui::SliderInt(label, &value, 0, 1, types[value], ImGuiSliderFlags_NoInput);
	const auto result = ImGui::IsItemFocused() ? isEdited : ImGui::IsItemClicked(ImGuiMouseButton_Left);

	ImGui::PopStyleColor(7);
	ImGui::PopStyleVar(3);
	return result;
}

auto ui::Selectable(const char* label, bool selected, ImGuiSelectableFlags flags, const Vector2& size) -> bool
{
	ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, Vector2{ 8.0f, 8.0f });
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
	if (isActive or selected)
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
		ImGui::PushStyleColor(ImGuiCol_Header, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);


		result = ImGui::Selectable(label, selected, flags, size);

		ImGui::PopStyleColor(4);
		break;
	case ItemState::hovered:
		ImGui::PushStyleColor(ImGuiCol_Header, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_HeaderActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);

		result = ImGui::Selectable(label, selected, flags, size);

		ImGui::PopStyleColor(5);
		break;
	case ItemState::pressed:
		{

			ImGui::PushStyleColor(ImGuiCol_Header, brush.controlFillColorDefaultBrush);
			ImGui::PushStyleColor(ImGuiCol_HeaderActive, brush.controlFillColorTertiaryBrush);
			ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brush.controlFillColorSecondaryBrush);
			ImGui::PushStyleColor(ImGuiCol_Border, brush.textControlElevationBorderFocusedBrush);
			ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);
			auto& drawList = *ImGui::GetWindowDrawList();
			const auto height = size.y == 0.0f ? ImGui::GetTextLineHeight() : size.y;
			const auto position = Vector2{ ImGui::GetCursorScreenPos() };


			result = ImGui::Selectable(label, selected, flags, size);

			drawList.AddRectFilled(position + Vector2{ 0.0f, 0.0f }, position + Vector2{ 4.0f, height },
								   brush.textControlElevationBorderFocusedBrush, 2.0f);

			ImGui::PopStyleColor(5);
		}
		break;
	case ItemState::disabled:
		ImGui::PushStyleColor(ImGuiCol_Header, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrokeColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);

		result = ImGui::Selectable(label, selected, flags, size);

		ImGui::PopStyleColor(3);
		break;
	}

	ImGui::PopStyleVar(2);

	return result;
}

auto ui::InputText(const char* label, const char* hint, std::string* text, ImGuiInputTextFlags flags) -> bool
{
	const auto& brush = ApplicationContext::getStyleBrush();
	const auto itemId = ImGui::GetID(label);
	const auto isHovered =
		ImGui::GetHoveredID() == itemId or ImGui::GetCurrentContext()->HoveredIdPreviousFrame == itemId;
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
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
		}
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorSecondaryBrush);
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorTertiaryBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorSecondaryBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDefaultBrush);
			}
		}
	}

	const auto position = Vector2{ ImGui::GetCursorScreenPos() };
	// ImGui::PushItemWidth(-1);
	const auto isEdited = ImGui::InputTextWithHint(label, hint, text, flags);
	// ImGui::PopItemWidth();
	const auto inputBoxSize = Vector2{ ImGui::GetItemRectSize() };

	auto& drawList = *ImGui::GetWindowDrawList();
	drawList.AddLine(
		position + Vector2{ 0.0f, inputBoxSize.y - borderSize }, position + inputBoxSize - Vector2{ 0.0f, borderSize },
		isActive ? brush.accentTextFillColorTertiaryBrush : brush.controlStrongStrokeColorDefaultBrush, borderSize);

	ImGui::PopStyleColor(5);
	ImGui::PopStyleVar(2);
	return isEdited;
}

auto ui::InputText(const char* label, const char* hint, char* buf, size_t buf_size, ImGuiInputTextFlags flags) -> bool
{
	const auto& brush = ApplicationContext::getStyleBrush();
	const auto itemId = ImGui::GetID(label);
	const auto isHovered =
		ImGui::GetHoveredID() == itemId or ImGui::GetCurrentContext()->HoveredIdPreviousFrame == itemId;
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
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
		}
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorSecondaryBrush);
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorTertiaryBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorSecondaryBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDefaultBrush);
			}
		}
	}

	const auto position = Vector2{ ImGui::GetCursorScreenPos() };
	// ImGui::PushItemWidth(-1);
	const auto isEdited = ImGui::InputTextWithHint(label, hint, buf, buf_size, flags);
	// ImGui::PopItemWidth();
	const auto inputBoxSize = Vector2{ ImGui::GetItemRectSize() };

	auto& drawList = *ImGui::GetWindowDrawList();
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
	HeadedTextOnly(header);
	const auto result = ui::InputText(label, nullptr, text, flags);

	return result;
}

auto ui::Combo(const char* label, int* current_item, const char* const items[], int items_count) -> bool
{
	const auto& brush = ApplicationContext::getStyleBrush();
	const auto itemId = ImGui::GetID(label);
	const auto isHovered =
		ImGui::GetHoveredID() == itemId or ImGui::GetCurrentContext()->HoveredIdPreviousFrame == itemId;
	const auto isActive = ImGui::GetActiveID() == itemId;
	const auto isDisabled = ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled;

	const auto borderSize = 2.0f;
	ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, borderSize);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
	ImGui::PushStyleColor(ImGuiCol_Border, brush.accentControlElevationBorderBrush);
	ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, brush.accentFillColorSelectedTextBackgroundBrush);

	if (isDisabled)
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorDisabledBrush);
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
		}
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.accentFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.accentFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorTertiaryBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorSecondaryBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDefaultBrush);
			}
		}
	}
	const auto position = Vector2{ ImGui::GetCursorScreenPos() };

	const auto isEdited = ImGui::Combo(label, current_item, items, items_count);

	const auto inputBoxSize = Vector2{ ImGui::GetItemRectSize() };

	if (not isDisabled)
	{
		auto& drawList = *ImGui::GetWindowDrawList();
		drawList.AddLine(position + Vector2{ borderSize, -borderSize },
						 position + Vector2{ inputBoxSize.x - borderSize, borderSize },
						 isHovered ? brush.accentFillColorDefaultBrush : brush.accentFillColorTertiaryBrush,
						 borderSize);
	}
	ImGui::PopStyleColor(8);
	ImGui::PopStyleVar(2);
	return isEdited;
}

auto ui::HeadedCombo(const std::string& header, const char* label, int* current_item, const char* const items[],
					 int items_count) -> bool
{
	HeadedTextOnly(header);
	const auto result = ui::Combo(label, current_item, items, items_count);
	return result;
}

auto ui::HeadedTextOnly(const std::string& header) -> void
{
	const auto& brush = ApplicationContext::getStyleBrush();
	const auto isDisabled = ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled;
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vector2{ 12.0f, 8.0f });
	if (isDisabled)
	{
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorDisabledBrush);
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);
	}
	ImGui::Text(header.c_str());
	ImGui::PopStyleColor();
	ImGui::PopStyleVar();
}

auto ui::DragFloat(const char* label, float* v, float v_speed, float v_min, float v_max, const char* format,
				   ImGuiSliderFlags flags) -> bool
{
	const auto& brush = ApplicationContext::getStyleBrush();
	const auto itemId = ImGui::GetID(label);
	const auto isHovered =
		ImGui::GetHoveredID() == itemId or ImGui::GetCurrentContext()->HoveredIdPreviousFrame == itemId;
	const auto isActive = ImGui::GetActiveID() == itemId;
	const auto isDisabled = ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled;

	const auto borderSize = 2.0f;
	ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, borderSize);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
	ImGui::PushStyleColor(ImGuiCol_Border, brush.accentControlElevationBorderBrush);
	ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, brush.accentFillColorSelectedTextBackgroundBrush);

	if (isDisabled)
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorDisabledBrush);
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
		}
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.accentFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.accentFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorTertiaryBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorSecondaryBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDefaultBrush);
			}
		}
	}
	const auto result = ImGui::DragFloat(label, v, v_speed, v_min, v_max, format, flags);
	ImGui::PopStyleColor(8);
	ImGui::PopStyleVar(2);
	return result;
}

auto ui::DragInt(const char* label, int* v, float v_speed, int v_min, int v_max, const char* format,
				 ImGuiSliderFlags flags) -> bool
{
	const auto& brush = ApplicationContext::getStyleBrush();
	const auto itemId = ImGui::GetID(label);
	const auto isHovered =
		ImGui::GetHoveredID() == itemId or ImGui::GetCurrentContext()->HoveredIdPreviousFrame == itemId;
	const auto isActive = ImGui::GetActiveID() == itemId;
	const auto isDisabled = ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled;

	const auto borderSize = 2.0f;
	ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, borderSize);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
	ImGui::PushStyleColor(ImGuiCol_Border, brush.accentControlElevationBorderBrush);
	ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, brush.accentFillColorSelectedTextBackgroundBrush);

	if (isDisabled)
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorDisabledBrush);
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
		}
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.accentFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.accentFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorTertiaryBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorSecondaryBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDefaultBrush);
			}
		}
	}
	const auto result = ImGui::DragInt(label, v, v_speed, v_min, v_max, format, flags);
	ImGui::PopStyleColor(8);
	ImGui::PopStyleVar(2);
	return result;
}

auto ui::HeadedDragInt(const std::string& header, const char* label, int* v, float v_speed, int v_min, int v_max,
					   const char* format, ImGuiSliderFlags flags) -> bool
{
	HeadedTextOnly(header);
	const auto result = ui::DragInt(label, v, v_speed, v_min, v_max, format, flags);

	return result;
}

auto ui::HeadedDragFloat(const std::string& header, const char* label, float* v, float v_speed, float v_min,
						 float v_max, const char* format, ImGuiSliderFlags flags) -> bool
{
	HeadedTextOnly(header);
	const auto result = ui::DragFloat(label, v, v_speed, v_min, v_max, format, flags);

	return result;
}

auto ui::HeadedDragInt3(const std::string& header, const char* label, int v[3], float v_speed, int v_min, int v_max,
						const char* format, ImGuiSliderFlags flags) -> bool
{
	HeadedTextOnly(header);
	const auto result = ui::DragInt3(label, v, v_speed, v_min, v_max, format, flags);

	return result;
}

auto ui::DragInt3(const char* label, int v[3], float v_speed, int v_min, int v_max, const char* format,
				  ImGuiSliderFlags flags) -> bool
{
	const auto& brush = ApplicationContext::getStyleBrush();
	const auto itemId = ImGui::GetID(label);
	const auto isHovered =
		ImGui::GetHoveredID() == itemId or ImGui::GetCurrentContext()->HoveredIdPreviousFrame == itemId;
	const auto isActive = ImGui::GetActiveID() == itemId;
	const auto isDisabled = ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled;

	const auto borderSize = 2.0f;
	ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, borderSize);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
	ImGui::PushStyleColor(ImGuiCol_Border, brush.accentControlElevationBorderBrush);
	ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, brush.accentFillColorSelectedTextBackgroundBrush);

	if (isDisabled)
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorDisabledBrush);
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDisabledBrush);
			}
		}
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.accentFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.accentFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);
		if (isActive)
		{
			ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorTertiaryBrush);
		}
		else
		{
			if (isHovered)
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorSecondaryBrush);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.controlFillColorDefaultBrush);
			}
		}
	}

	ImGuiWindow* window = ImGui::GetCurrentWindow();
	if (window->SkipItems)
		return false;
	const auto components = 3;

	auto& drawList = *ImGui::GetWindowDrawList();
	const auto red = isDisabled ? brush.controlFillColorDisabledBrush : Color{ 0.84f, 0.16f, 0.16f };
	const auto green = isDisabled ? brush.controlFillColorDisabledBrush : Color{ .31f, 0.47f, 0.18f };
	const auto blue = isDisabled ? brush.controlFillColorDisabledBrush : Color{ 0.26f, 0.38f, 0.93f };

	bool value_changed = false;
	ImGui::BeginGroup();
	ImGui::PushID(label);
	ImGui::PushMultiItemsWidths(components, ImGui::CalcItemWidth());

	{
		ImGui::PushID(0);
		ImGui::PushStyleColor(ImGuiCol_Border, red);
		value_changed |= ImGui::DragScalar("", ImGuiDataType_S32, &v[0], v_speed, &v_min, &v_max, format, flags);
		ImGui::PopStyleColor();
		ImGui::SameLine(0, 0);

		const auto min = Vector2{ ImGui::GetItemRectMax().x - 4, ImGui::GetItemRectMin().y };
		const auto max =
			Vector2{ ImGui::GetItemRectMax().x + ImGui::CalcTextSize("X").x + 4.0f, ImGui::GetItemRectMax().y };


		drawList.AddRectFilled(min, max, red, 4,
							   ImDrawFlags_RoundCornersTopRight | ImDrawFlags_RoundCornersBottomRight);

		ImGui::Text("X");
		ImGui::PopID();
		ImGui::PopItemWidth();
	}
	ImGui::SameLine();
	{
		ImGui::PushID(1);
		ImGui::PushStyleColor(ImGuiCol_Border, green);
		value_changed |= ImGui::DragScalar("", ImGuiDataType_S32, &v[1], v_speed, &v_min, &v_max, format, flags);
		ImGui::PopStyleColor();
		ImGui::SameLine(0, 0);

		const auto min = Vector2{ ImGui::GetItemRectMax().x - 4, ImGui::GetItemRectMin().y };
		const auto max =
			Vector2{ ImGui::GetItemRectMax().x + ImGui::CalcTextSize("Y").x + 4.0f, ImGui::GetItemRectMax().y };


		drawList.AddRectFilled(min, max, green, 4,
							   ImDrawFlags_RoundCornersTopRight | ImDrawFlags_RoundCornersBottomRight);
		ImGui::Text("Y");
		ImGui::PopID();
		ImGui::PopItemWidth();
	}
	ImGui::SameLine();
	{
		ImGui::PushID(2);
		ImGui::PushStyleColor(ImGuiCol_Border, blue);
		value_changed |= ImGui::DragScalar("", ImGuiDataType_S32, &v[2], v_speed, &v_min, &v_max, format, flags);
		ImGui::PopStyleColor();
		ImGui::SameLine(0, 0);

		const auto min = Vector2{ ImGui::GetItemRectMax().x - 4, ImGui::GetItemRectMin().y };
		const auto max =
			Vector2{ ImGui::GetItemRectMax().x + ImGui::CalcTextSize("Z").x + 4.0f, ImGui::GetItemRectMax().y };


		drawList.AddRectFilled(min, max, blue, 4,
							   ImDrawFlags_RoundCornersTopRight | ImDrawFlags_RoundCornersBottomRight);
		ImGui::Text("Z");
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	ImGui::PopID();

	ImGui::EndGroup();

	ImGui::PopStyleColor(8);
	ImGui::PopStyleVar(2);
	return value_changed;
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

	brush.controlFillColorDefaultBrush = Color{ 0.2f, 0.2f, 0.2f, 1.0f };
	brush.controlFillColorSecondaryBrush = Color{ 0.24f, 0.24f, 0.24f, 1.0f };
	brush.controlFillColorTertiaryBrush = Color{ 0.15f, 0.15f, 0.15f, 1.0f };
	brush.controlFillColorDisabledBrush = Color{ 0.16f, 0.16f, 0.16f, 1.0f };
	brush.controlFillColorTransparentBrush = Color{ 0.13f, 0.13f, 0.13f, 1.0f };
	brush.controlFillColorInputActiveBrush = Color{ 0.12f, 0.12f, 0.12f, 1.0f };

	brush.controlStrokeColorDefaultBrush = Color{ 0.19f, 0.19f, 0.19f, 1.0f };
	brush.controlStrokeColorSecondaryBrush = Color{ 0.24f, 0.24f, 0.24f, 1.0f };
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

	brush.acrylicBackgroundFillColorBaseBrush = Color{ 0.13f, 0.13f, 0.13f, 1.0f };
	brush.acrylicBackgroundFillColorDefaultBrush = Color{ 0.18f, 0.18f, 0.18f, 1.0f };

	brush.systemFillColorSuccessBrush = Color{ 0.42f, 0.8f, 0.37f, 1.0f };
	brush.systemFillColorCautionBrush = Color{ 0.99f, 0.88f, 0.0f, 1.0f };
	brush.systemFillColorCriticalBrush = Color{ 1.0f, 0.6f, 0.64f, 1.0f };
	brush.systemFillColorSuccessBackgroundBrush = Color{ 0.22f, 0.24f, 0.11f, 1.0f };
	brush.systemFillColorCautionBackgroundBrush = Color{ 0.26f, 0.21f, 0.1f, 1.0f };
	brush.systemFillColorCriticalBackgroundBrush = Color{ 0.27f, 0.15f, 0.15f, 1.0f };

	return brush;
}
