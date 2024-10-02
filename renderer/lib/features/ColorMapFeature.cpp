#include "ColorMapFeature.h"
#include "Logging.h"
#include "owl/helper/cuda.h"

void b3d::renderer::ColorMapFeature::onInitialize()
{
	colorMapTexture_ = sharedParameters_->get<ExternalTexture>("colorMapTexture");
	coloringInfo_ = sharedParameters_->get<ColoringInfo>("coloringInfo");
	colorMapInfos_ = sharedParameters_->get<ColorMapInfos>("colorMapInfos");
}

auto b3d::renderer::ColorMapFeature::beginUpdate() -> void
{
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
		texDesc.sRGB = 1;

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
auto b3d::renderer::ColorMapFeature::getParamsData() -> ParamsData
{
	assert(coloringInfo_ != nullptr && colorMapInfos_ != nullptr);
	return { colorMapCudaTexture_, coloringInfo_->singleColor, coloringInfo_->selectedColorMap,
			 coloringInfo_->coloringMode };
}
