#include "FontCollection.h"
#include <IconsFontAwesome6Brands.h>
#include <IconsLucide.h>

#include <imgui.h>

auto FontCollection::rebuildFont(const std::vector<float>& dpiList) -> void
{
	auto& io = ImGui::GetIO();

	io.Fonts->ClearFonts();
	loadedFonts_.clear();

	constexpr auto baseFontSize = 16.0f;
	constexpr auto titleFontSize = 24.0f;
	auto config = ImFontConfig{};

	config.OversampleH = 8;
	config.OversampleV = 8;

	for (auto dpi : dpiList)
	{
		if (!dpiToFont_.contains(dpi))
		{
			config.SizePixels = titleFontSize * dpi;
			auto fontBold = io.Fonts->AddFontFromFileTTF("resources/fonts/Roboto-Bold.ttf", config.SizePixels, &config);
			config.SizePixels = dpi * baseFontSize;
			auto font = io.Fonts->AddFontFromFileTTF("resources/fonts/Roboto-Medium.ttf", config.SizePixels, &config);


			static auto iconRangesLucide = ImVector<ImWchar>{};
			ImFontGlyphRangesBuilder builder;
			builder.AddText(
				ICON_LC_ROTATE_3D ICON_LC_MOVE_3D ICON_LC_SCALE_3D ICON_LC_BAR_CHART_3 ICON_LC_UNPLUG ICON_LC_LOG_OUT
					ICON_LC_CIRCLE_GAUGE ICON_LC_BUG ICON_LC_SERVER ICON_LC_SERVER_COG ICON_LC_SERVER_CRASH
						ICON_LC_SERVER_OFF ICON_LC_CIRCLE_CHECK ICON_LC_HARD_DRIVE ICON_LC_TRIANGLE_ALERT
							ICON_LC_ARROW_RIGHT_LEFT ICON_LC_REFRESH_CW ICON_LC_BOX ICON_LC_VIEW
								ICON_LC_SCROLL_TEXT /*ICON_LC_EYE*/ ICON_LC_X ICON_LC_UNDO_2 ICON_LC_FILE_UP
									ICON_LC_TRASH_2 ICON_LC_PENCIL ICON_LC_FOLDER ICON_LC_ROTATE_CCW ICON_LC_SEARCH);
			builder.BuildRanges(&iconRangesLucide);

			const auto iconFontSize = dpi * baseFontSize * 2.0f / 3.0f;
			config.MergeMode = true;
			config.PixelSnapH = true;
			config.GlyphMinAdvanceX = iconFontSize;
			config.OversampleH = 8;
			config.OversampleV = 8;

			font = io.Fonts->AddFontFromFileTTF("resources/fonts/lucide.ttf", iconFontSize, &config,
												iconRangesLucide.Data);

			static auto iconRangesFontAwesomeBrands = ImVector<ImWchar>{};
			builder.Clear();
			builder.AddText(ICON_FA_GITHUB);
			builder.BuildRanges(&iconRangesFontAwesomeBrands);

			font = io.Fonts->AddFontFromFileTTF("resources/fonts/fa-brands-400.ttf", iconFontSize, &config,
												iconRangesFontAwesomeBrands.Data);

			config.GlyphMinAdvanceX = iconFontSize * 2.0f;
			config.MergeMode = false;
			auto fontBig = io.Fonts->AddFontFromFileTTF("resources/fonts/lucide.ttf", iconFontSize * 2.0f, &config,
														iconRangesLucide.Data);

			const auto fontIndex = loadedFonts_.size();
			loadedFonts_.push_back(font);
			loadedFonts_.push_back(fontBig);
			loadedFonts_.push_back(fontBold);
			dpiToFont_[dpi] = static_cast<int>(fontIndex);
		}
	}

	{

		config.SizePixels = 8 * baseFontSize;
		static auto gpuCpuText = ImVector<ImWchar>{};
		ImFontGlyphRangesBuilder builder;
		builder.Clear();
		builder.AddText("GCPU");
		builder.BuildRanges(&gpuCpuText);


		gpuCpuExtraBigTextFont_ = io.Fonts->AddFontFromFileTTF("resources/fonts/Roboto-Medium.ttf", config.SizePixels,
															   &config, gpuCpuText.Data);
	}
}

auto FontCollection::containsDpi(const float dpi) const noexcept -> bool
{
	return dpiToFont_.contains(dpi);
}

auto FontCollection::getDefaultFont() const noexcept -> ImFont*
{
	assert(loadedFonts_.size() > 0);
	return loadedFonts_[currentFontIndex_];
}

auto FontCollection::getTitleFont() const noexcept -> ImFont*
{
	assert(loadedFonts_.size() > (currentFontIndex_ + 2));
	return loadedFonts_[currentFontIndex_ + 2];
}

auto FontCollection::getBigIconsFont() const noexcept -> ImFont*
{
	assert(loadedFonts_.size() > (currentFontIndex_ + 1));
	return loadedFonts_[currentFontIndex_ + 1]; // TODO: needs better indexing
}

auto FontCollection::getDefaultFontDpiScale() const noexcept -> float
{
	return getDefaultFont()->Scale;
	;
}
auto FontCollection::getGpuCpuExtraBigTextFont() const noexcept -> ImFont*
{
	return gpuCpuExtraBigTextFont_;
}
