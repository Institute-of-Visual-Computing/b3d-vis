#include "ColorMapFeature.h"

#include <format>
#include "Logging.h"

#include "imgui.h"
#include "imgui_internal.h"

#include "owl/helper/cuda.h"

auto b3d::renderer::ColorMapFeature::beginUpdate() -> void
{
	colorMapTexture_ = sharedParameters_->get<ExternalTexture>("colorMapTexture");
	coloringInfo_ = sharedParameters_->get<ColoringInfo>("coloringInfo");
	colorMapInfos_ = sharedParameters_->get<ColorMapInfos>("colorMapInfos");

	skipUpdate = colorMapTexture_ == nullptr || coloringInfo_ == nullptr || colorMapInfos_ == nullptr;

	if (skipUpdate)
	{
		b3d::renderer::log("ColorMapFeature skips update, because of missing shared parameters!");
		return;
	}

	cudaArray_t colorMapCudaArray{};
	{
		OWL_CUDA_CHECK(cudaGraphicsMapResources(1, &colorMapTexture_->target));

		OWL_CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&colorMapCudaArray, colorMapTexture_->target, 0, 0));

		// Create texture
		auto resDesc = cudaResourceDesc{};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = colorMapCudaArray;

		auto texDesc = cudaTextureDesc{};
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;

		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType; // cudaReadModeNormalizedFloat

		texDesc.normalizedCoords = 1;
		texDesc.maxAnisotropy = 1;
		texDesc.maxMipmapLevelClamp = 0;
		texDesc.minMipmapLevelClamp = 0;
		texDesc.mipmapFilterMode = cudaFilterModePoint;
		texDesc.borderColor[0] = 1.0f;
		texDesc.borderColor[1] = 1.0f;
		texDesc.borderColor[2] = 1.0f;
		texDesc.borderColor[3] = 1.0f;
		texDesc.sRGB = 0;

		OWL_CUDA_CHECK(cudaCreateTextureObject(&colorMapCudaTexture_, &resDesc, &texDesc, nullptr));
	}
}
auto b3d::renderer::ColorMapFeature::endUpdate() -> void
{
	if (skipUpdate)
	{
		return;
	}

	{
		OWL_CUDA_CHECK(cudaDestroyTextureObject(colorMapCudaTexture_));
		cudaGraphicsUnmapResources(1, &colorMapTexture_->target);
	}
}
auto b3d::renderer::ColorMapFeature::gui() -> void
{

	if (colorMapTexture_ != nullptr && colorMapTexture_->nativeHandle != nullptr)
	{
		const auto totalItems = colorMapInfos_->colorMapNames->size();

		ImGui::Combo("Mode", &selectedColoringMode_, "Uniform Color\0ColorMap\0\0");
		if (selectedColoringMode_ == 0)
		{
			ImGui::ColorEdit3("Uniform Color", &coloringInfo_->singleColor.x);
		}
		else
		{

			ImGui::SetNextItemWidth(-1);
			if (ImGui::BeginCombo("##coloringModeSelector", "", ImGuiComboFlags_CustomPreview))
			{
				const auto size = ImGui::GetContentRegionAvail();
				const auto mapItemSize = ImVec2{ size.x, 20 };
				ImGui::Image(colorMapTexture_->nativeHandle, mapItemSize,
							 ImVec2(0, (selectedColoringMap_ + 0.5) / static_cast<float>(totalItems)),
							 ImVec2(1, (selectedColoringMap_ + 0.5) / static_cast<float>(totalItems)));

				for (auto n = 0; n < colorMapInfos_->colorMapNames->size(); n++)
				{
					const auto isSelected = (selectedColoringMap_ == n);
					if (ImGui::Selectable(std::format("##colorMap{}", n).c_str(), isSelected,
										  ImGuiSelectableFlags_AllowOverlap, mapItemSize))
					{
						selectedColoringMap_ = n;
					}
					ImGui::SameLine(1);
					ImGui::Image(colorMapTexture_->nativeHandle, mapItemSize,
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
				const auto size = ImGui::GetContentRegionAvail();
				const auto mapItemSize = ImVec2{ size.x, 20 };
				ImGui::Image(colorMapTexture_->nativeHandle, mapItemSize,
							 ImVec2(0, selectedColoringMap_ / static_cast<float>(totalItems)),
							 ImVec2(1, (selectedColoringMap_ + 1) / static_cast<float>(totalItems)));
				ImGui::EndComboPreview();
			}
		}
	}
}
auto b3d::renderer::ColorMapFeature::getParamsData() -> ParamsData
{
	assert(coloringInfo_ != nullptr && colorMapInfos_ != nullptr);
	coloringInfo_->coloringMode = selectedColoringMode_ == 0 ? single : colormap;
	coloringInfo_->selectedColorMap = colorMapInfos_->firstColorMapYTextureCoordinate +
		static_cast<float>(selectedColoringMap_) * colorMapInfos_->colorMapHeightNormalized;

	return { colorMapCudaTexture_, coloringInfo_->singleColor, coloringInfo_->selectedColorMap,
			 coloringInfo_->coloringMode };
}
