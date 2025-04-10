#include "Style.h"



#include "framework/ApplicationContext.h"

auto DecoratedButton(const char* label, const Vector2& size, bool disabled) -> bool
{
	const auto& brush = ApplicationContext::getStyleBrush();

	const auto itemId = ImGui::GetID(label);
	const auto isHovered = ImGui::GetHoveredID() == itemId;
	const auto isActive = ImGui::GetActiveID() == itemId;


	if (disabled)
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.controlFillColorDisabledBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.cardStrokeColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorDisabledBrush);
	}
	else if (isHovered)
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);
	}
	else if (isActive)
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.cardStrokeColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorSecondaryBrush);
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_Button, brush.controlFillColorDefaultBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, brush.controlFillColorSecondaryBrush);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, brush.controlFillColorTertiaryBrush);
		ImGui::PushStyleColor(ImGuiCol_Border, brush.controlElevationBorderBrush);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.textFillColorPrimaryBrush);
	}

	const auto result = ImGui::Button(label, size);

	ImGui::PopStyleColor(5);
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

	brush.controlElevationBorderBrush = Color{ 0.19f, 0.19f, 0.19f, 1.0f };

	brush.cardStrokeColorDefaultBrush = Color{ 0.11f, 0.11f, 0.11f, 1.0f };
	brush.cardStrokeColorDefaultSolidBrush = Color{ 0.11f, 0.11f, 0.11f, 1.0f };

	return brush;
}
