#pragma once

#include <unordered_map>
#include <vector>

struct ImFont;

class FontCollection
{
public:
	auto rebuildFont(const std::vector<float>& dpiList) -> void;

	[[nodiscard]] auto containsDpi(const float dpi) const noexcept -> bool;

	[[nodiscard]] auto getDefaultFont() const noexcept -> ImFont*;
	[[nodiscard]] auto getBigIconsFont() const noexcept -> ImFont*;
	[[nodiscard]] auto getDefaultFontDpiScale() const noexcept -> float;
	[[nodiscard]] auto getGpuCpuExtraBigTextFont() const noexcept -> ImFont*;


private:
	std::vector<ImFont*> loadedFonts_;
	std::unordered_map<float, int> dpiToFont_{};
	int currentFontIndex_{ 0 };
	ImFont* gpuCpuExtraBigTextFont_{};
};
