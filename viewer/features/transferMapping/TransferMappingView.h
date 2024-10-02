#pragma once

#include "framework/DockableWindowViewBase.h"

#include <owl/common.h>

#include <string>
#include <vector>

class TransferMappingView final : public DockableWindowViewBase
{
public:
	TransferMappingView(ApplicationContext& appContext, Dockspace* dockspace);

	[[nodiscard]] auto resampleData(int samplesCount) const -> std::vector<float>;

	auto hasNewDataAvailable() const -> bool
	{
		return newDataAvailable_;
	}

	auto setColorMapInfos(const std::vector<std::string>& names, void* colorMapTextureHandle) -> void
	{
		colorMapNames_ = names;
		colorMapTextureHandle_ = colorMapTextureHandle;
	}

	[[nodiscard]] auto getColoringMode() const -> auto
	{
		return selectedColoringMode_;
	}

	[[nodiscard]] auto getColoringMap() const -> auto
	{
		return selectedColoringMap_;
	}

private:
	auto onDraw() -> void override;
	std::vector<ImVec2> dataPoints_;


	int selectedCurveHandleIdx_{ -1 };
	bool newDataAvailable_{ true };


	std::vector<std::string> colorMapNames_{};
	void* colorMapTextureHandle_{ nullptr };
	int selectedColoringMode_{ 1 };
	int selectedColoringMap_{ 0 };
	owl::vec4f uniformColor_{ 0.0f, 0.0f, 0.0f, 1.0f };
};
