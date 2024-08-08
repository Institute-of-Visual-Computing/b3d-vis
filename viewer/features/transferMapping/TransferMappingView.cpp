#define IMGUI_DEFINE_MATH_OPERATORS
#include "TransferMappingView.h"


#include <Curve.h>

#include <imgui_internal.h>
#include "imgui.h"

TransferMappingView::TransferMappingView(ApplicationContext& appContext, Dockspace* dockspace)
	: DockableWindowViewBase{ appContext, "Transfer Mapping", dockspace, WindowFlagBits::none }
{
	dataPoints_.resize(10);
	dataPoints_[0] = { 0.00000000f, 0.00000000f };
	dataPoints_[1] = { 0.00158982514f, 0.420000017f };
	dataPoints_[2] = { 0.0810810775f, 0.779999971f };
	dataPoints_[3] = { 0.270270258f, 0.845000029f };
	dataPoints_[4] = { 0.511923671f, 1.00000000f };
	dataPoints_[5] = { 0.686804473f, 0.654999971f };
	dataPoints_[6] = { 0.812400639f, 0.754999995f };
	dataPoints_[7] = { 1.00000000f, 0.605000019f };
	dataPoints_[8].x = ImGui::CurveTerminator;
}

auto TransferMappingView::resampleData(const int samplesCount) const -> std::vector<float>
{
	auto samples = std::vector<float>{};
	samples.resize(samplesCount);

	const auto inc = 1.0f / (samples.size() - 1);
	for (auto i = 0; i < samples.size(); i++)
	{
		samples[i] = ImGui::CurveValue(i * inc, dataPoints_.size(), dataPoints_.data());
	}
	return samples;
}

auto TransferMappingView::onDraw() -> void
{
	const auto availableSize = ImGui::GetContentRegionAvail();
	const auto size = ImVec2{ availableSize.x, std::min({ 200.0f, availableSize.y }) };
	newDataAvailable_ = false;

	{
		const auto totalItems = colorMapNames_.size();
		assert(colorMapTextureHandle_);

		ImGui::Combo("Mode", &selectedColoringMode_, "Uniform Color\0ColorMap\0\0");
		if (selectedColoringMode_ == 0)
		{
			if (ImGui::ColorEdit3("Uniform Color", &uniformColor_.x))
			{
				newDataAvailable_ = true;
			}
		}
		else
		{
			ImGui::SetNextItemWidth(-1);
			if (ImGui::BeginCombo("##coloringModeSelector", "", ImGuiComboFlags_CustomPreview))
			{
				const auto mapItemSize = ImVec2{ ImGui::GetContentRegionAvail().x, 20 };
				ImGui::Image(colorMapTextureHandle_, mapItemSize,
							 ImVec2(0, (selectedColoringMap_ + 0.5) / static_cast<float>(totalItems)),
							 ImVec2(1, (selectedColoringMap_ + 0.5) / static_cast<float>(totalItems)));

				for (auto n = 0; n < totalItems; n++)
				{
					const auto isSelected = (selectedColoringMap_ == n);
					if (ImGui::Selectable(std::format("##colorMap{}", n).c_str(), isSelected,
										  ImGuiSelectableFlags_AllowOverlap, mapItemSize))
					{
						newDataAvailable_ = true;
						selectedColoringMap_ = n;
					}
					ImGui::SameLine(1);
					ImGui::Image(colorMapTextureHandle_, mapItemSize,
								 ImVec2(0, (n + 0.5) / static_cast<float>(totalItems)),
								 ImVec2(1, (n + 0.5) / static_cast<float>(totalItems)));

					if (isSelected)
					{
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}

			if (ImGui::BeginComboPreview())
			{
				const auto mapItemSize = ImVec2{ ImGui::GetContentRegionAvail().x, 20 };
				ImGui::Image(colorMapTextureHandle_, mapItemSize,
							 ImVec2(0, selectedColoringMap_ / static_cast<float>(totalItems)),
							 ImVec2(1, (selectedColoringMap_ + 1) / static_cast<float>(totalItems)));
				ImGui::EndComboPreview();
			}
		}
	}



	// TODO:: Curve crashes sometimes in release
	if (ImGui::Curve("##transferFunction", size, dataPoints_.size(), dataPoints_.data(), &selectedCurveHandleIdx_))
	{
		newDataAvailable_ = true;
	}
}
