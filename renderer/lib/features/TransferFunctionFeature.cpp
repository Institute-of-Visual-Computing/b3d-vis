#include "TransferFunctionFeature.h"

#include "Curve.h"

using namespace b3d::renderer;

TransferFunctionFeature::TransferFunctionFeature(const std::string& name, const size_t dataPointsCount)
	: RenderFeature{ name }, dataPoints_(dataPointsCount)
{
	assert(dataPointsCount > 0);
	dataPoints_[0].x = ImGui::CurveTerminator;
}

auto TransferFunctionFeature::beginUpdate() -> void
{
	//TODO: check all shared params and skip if they are incomplete
	skipUpdate = false;

	if (skipUpdate)
	{
		return;
	}
}

auto TransferFunctionFeature::endUpdate() -> void
{
	if (skipUpdate)
	{
		return;
	}
}
auto TransferFunctionFeature::gui() -> void
{
	const auto availableSize = ImGui::GetContentRegionAvail();
	const auto size = ImVec2{ availableSize.x , std::min( {200.0f, availableSize.y}) };

    if (ImGui::Curve("##transferFunction", size, dataPoints_.size(), dataPoints_.data(), &selectedCurveHandleIdx_))
    {
        // curve changed
		//TODO: maybe trigger an event to refit/recreate data
    }

	//TODO: resample to a appropriate size
    const auto resampledValue = ImGui::CurveValue(0.7f, dataPoints_.size(), dataPoints_.data()); // calculate value at position 0.7
}
