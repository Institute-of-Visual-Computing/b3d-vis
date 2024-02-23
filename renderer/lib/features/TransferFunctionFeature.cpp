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
	transferFunctionBuffer_ = sharedParameters_->get<ExternalBuffer>("transferFunctionBuffer");

	skipUpdate = transferFunctionBuffer_ == nullptr;

	if (skipUpdate)
	{
		return;
	}

	void* devPtr = nullptr;
	size_t cudaBfrSize{};
	{
		OWL_CUDA_CHECK(cudaGraphicsMapResources(1, &transferFunctionBuffer_->target));

		cudaGraphicsResourceGetMappedPointer(&devPtr, &cudaBfrSize, transferFunctionBuffer_->target);
		// Create texture
		auto resDesc = cudaResourceDesc{};
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = devPtr;
		resDesc.res.linear.sizeInBytes = cudaBfrSize;
		resDesc.res.linear.desc = cudaCreateChannelDesc(sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat);
		
		auto texDesc = cudaTextureDesc{};
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;

		texDesc.filterMode = cudaFilterModeLinear;
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

		OWL_CUDA_CHECK(cudaCreateTextureObject(&transferFunctionCudaTexture_, &resDesc, &texDesc, nullptr));
	}
}

auto TransferFunctionFeature::endUpdate() -> void
{
	if (skipUpdate)
	{
		return;
	}
	OWL_CUDA_CHECK(cudaDestroyTextureObject(transferFunctionCudaTexture_));
	cudaGraphicsUnmapResources(1, &transferFunctionBuffer_->target);

}

auto TransferFunctionFeature::gui() -> void
{
	const auto availableSize = ImGui::GetContentRegionAvail();
	const auto size = ImVec2{ availableSize.x, std::min({ 200.0f, availableSize.y }) };

	if (ImGui::Curve("##transferFunction", size, dataPoints_.size(), dataPoints_.data(), &selectedCurveHandleIdx_))
	{
		// curve changed
		// TODO: maybe trigger an event to refit/recreate data
	}

	// TODO: resample to a appropriate size
	const auto resampledValue =
		ImGui::CurveValue(0.7f, dataPoints_.size(), dataPoints_.data()); // calculate value at position 0.7
}
auto TransferFunctionFeature::getParamsData() -> ParamsData
{
	return { transferFunctionCudaTexture_ };
}
