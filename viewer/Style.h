#pragma once

#include <imgui.h>

#include "Color.h"
#include "Mathematics.h"

#ifdef WIN32
#include <winrt/Windows.UI.ViewManagement.h>
#endif
#include <imgui_internal.h>


struct AccentColors
{
	Color accent = Color{ 0.0332f, 0.19141f, 0.69531f };
	Color accentLight1 = Color{ 0.09082f, 0.33203f, 0.77734f };
	Color accentLight2 = Color{ 0.21582f, 0.48438f, 0.85547f };
	Color accentLight3 = Color{ 0.41797f, 0.6875f, 1.0f };
	Color accentDark1 = Color{ 0.01685f, 0.10205f, 0.34766f };
	Color accentDark2 = Color{ 0.00854f, 0.05469f, 0.18164f };
	Color accentDark3 = Color{ 0.00275f, 0.01941f, 0.05469f };

#ifdef WIN32
	constexpr AccentColors()
	{

		winrt::Windows::UI::ViewManagement::UISettings const ui_settings{};
		accent = Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::Accent) };
		accentLight1 =
			Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentLight1) };
		accentLight2 =
			Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentLight2) };
		accentLight3 =
			Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentLight3) };
		accentDark1 = Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentDark1) };
		accentDark2 = Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentDark2) };
		accentDark3 = Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentDark3) };
	}
#endif
};
// use winui3 brush names
struct StyleBrush
{
	Color textFillColorPrimaryBrush;
	Color textFillColorSecondaryBrush;
	Color textFillColorTertiaryBrush;
	Color textFillColorDisabledBrush;
	Color accentTextFillColorPrimaryBrush;
	Color accentTextFillColorSecondaryBrush;
	Color accentTextFillColorTertiaryBrush;
	Color accentTextFillColorDisabledBrush;
	Color textOnAccentFillColorPrimaryBrush;
	Color textOnAccentFillColorSecondaryBrush;
	Color textOnAccentFillColorDisabledBrush;
	Color textOnAccentFillColorSelectedTextBrush;

	Color cardBackgroundFillColorDefaultBrush;
	Color cardBackgroundFillColorSecondaryBrush;

	Color accentAcrylicBackgroundFillColorBaseBrush;
	Color accentAcrylicBackgroundFillColorDefaultBrush;

	Color solidBackgroundFillColorBaseBrush;
	Color solidBackgroundFillColorBaseAltBrush;
	Color solidBackgroundFillColorSecondaryBrush;
	Color solidBackgroundFillColorTertiaryBrush;
	Color solidBackgroundFillColorQuarternaryBrush;

	Color controlFillColorDefaultBrush;
	Color controlFillColorSecondaryBrush;
	Color controlFillColorTertiaryBrush;
	Color controlFillColorDisabledBrush;
	Color controlFillColorTransparentBrush;
	Color controlFillColorInputActiveBrush;

	Color controlElevationBorderBrush;

	Color cardStrokeColorDefaultBrush;
	Color cardStrokeColorDefaultSolidBrush;
};


auto DecoratedButton(const char* label, const Vector2& size = Vector2{}, bool disabled = false) -> bool;
auto createDarkThemeBrush(const AccentColors& accentColors) -> StyleBrush;
