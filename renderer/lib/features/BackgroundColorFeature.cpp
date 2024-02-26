#include "BackgroundColorFeature.h"

#include <format>

#include "imgui.h"

auto b3d::renderer::BackgroundColorFeature::getParamsData() -> ParamsData
{
	return { colors_ };
}

auto b3d::renderer::BackgroundColorFeature::gui() -> void
{
	/*
	for (auto i = 0; i < S; i++)
	{
		ImGui::ColorEdit3(std::format("Color {}", i).c_str(), &colors_[i].r);
	}
	*/
	ImGui::ColorEdit3("Color 1", &colors_[0].r);
	ImGui::ColorEdit3("Color 2", &colors_[1].r);
}
auto b3d::renderer::BackgroundColorFeature::hasGui() const -> bool
{
	return true;
}
