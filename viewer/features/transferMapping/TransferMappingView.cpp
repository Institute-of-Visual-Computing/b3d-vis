#include "TransferMappingView.h"
#include <Curve.h>

TransferMappingView::TransferMappingView(ApplicationContext& appContext, Dockspace* dockspace)
	: DockableWindowViewBase{ appContext, "Transfer Mapping", dockspace, WindowFlagBits::none }
{
	resizeDataPoints(10);
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

auto TransferMappingView::onDraw() -> void
{
	const auto availableSize = ImGui::GetContentRegionAvail();
	const auto size = ImVec2{ availableSize.x, std::min({ 200.0f, availableSize.y }) };

	// TODO:: Curve crashes sometimes in release
	if (ImGui::Curve("##transferFunction", size, dataPoints_.size(), dataPoints_.data(), &selectedCurveHandleIdx_))
	{
		newDataAvailable_ = true;
	}

	if (newDataAvailable_)
	{
		newDataAvailable_ = false;
		
		const auto inc = 1.0f / (stagingBuffer_.size() - 1);
		for (auto i = 0; i < stagingBuffer_.size(); i++)
		{
			stagingBuffer_[i] = ImGui::CurveValue(i * inc, dataPoints_.size(), dataPoints_.data());
		}
	}
}

auto TransferMappingView::resizeDataPoints(const int size) -> void
{
	stagingBuffer_.resize(size);
}
