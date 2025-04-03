#include "TransferFunctionFeature.h"
#include "Logging.h"
#include <owl/helper/cuda.h>

using namespace b3d::renderer;

TransferFunctionFeature::TransferFunctionFeature(const std::string& name, [[maybe_unused]] const size_t dataPointsCount)
	: RenderFeature{ name }, transferFunctionTexture_{ nullptr }
{
}

void TransferFunctionFeature::onInitialize()
{
	transferFunctionTexture_ = sharedParameters_->get<ExternalTexture>("transferFunctionTexture");
}

auto TransferFunctionFeature::beginUpdate() -> void
{
	skipUpdate = transferFunctionTexture_ == nullptr;

	if (skipUpdate)
	{
		b3d::renderer::log("TransferFunctionFeature skips update, because of missing shared parameters!");
		return;
	}
	
	OWL_CUDA_CHECK(cudaGraphicsMapResources(1, &transferFunctionTexture_->target));

	OWL_CUDA_CHECK(
		cudaGraphicsSubResourceGetMappedArray(&transferFunctionCudaArray_, transferFunctionTexture_->target, 0, 0));

	// Create texture
	auto resDesc = cudaResourceDesc{};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = transferFunctionCudaArray_;
	
	
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

auto TransferFunctionFeature::endUpdate() -> void
{
	if (skipUpdate)
	{
		return;
	}
	OWL_CUDA_CHECK(cudaDestroyTextureObject(transferFunctionCudaTexture_));
	OWL_CUDA_CHECK(cudaGraphicsUnmapResources(1, &transferFunctionTexture_->target));
}

auto TransferFunctionFeature::getParamsData() -> ParamsData
{
	return { transferFunctionCudaTexture_ };
}
