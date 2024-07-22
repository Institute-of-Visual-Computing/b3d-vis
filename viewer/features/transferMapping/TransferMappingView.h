#pragma once

#include "framework/DockableWindowViewBase.h"

#include <vector>

class TransferMappingView final : public DockableWindowViewBase
{
public:
	TransferMappingView(ApplicationContext& appContext, Dockspace* dockspace);

	auto onDraw() -> void override;

	auto getData() const -> const std::vector<float>&
	{
		return stagingBuffer_;
	}
	auto resizeDataPoints(const int size) -> void;
	auto newDataAvailable() const -> bool
	{
		return newDataAvailable_;
	}
	private:
	std::vector<ImVec2> dataPoints_;
	std::vector<float> stagingBuffer_;
	int selectedCurveHandleIdx_{ -1 };
	bool newDataAvailable_{ true };
};
