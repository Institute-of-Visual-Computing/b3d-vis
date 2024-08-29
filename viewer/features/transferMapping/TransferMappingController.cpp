
#include "TransferMappingController.h"
#include <cuda_runtime.h>
#include <owl/helper/cuda.h>
#include "TransferMapping.h"
#include "TransferMappingView.h"
#include "framework/ApplicationContext.h"

TransferMappingController::TransferMappingController(ApplicationContext& applicationContext,
													 TransferMapping& transferMapping)
	: UpdatableComponentBase(applicationContext),
	  mappingView_{ std::make_unique<TransferMappingView>(applicationContext, applicationContext.getMainDockspace()) },
	  transferMapping_{ &transferMapping }
{
	applicationContext.addMenuToggleAction(
		showToolWindow_, [&](const bool isOn) { isOn ? mappingView_->open() : mappingView_->close(); }, "Tools",
		"Transfer Mapping");
}

auto TransferMappingController::update() -> void
{
	showToolWindow_ = mappingView_->isOpen();
	if (showToolWindow_)
	{
		mappingView_->setColorMapInfos(transferMapping_->colorMapResources_.colorMap.colorMapNames,
									   reinterpret_cast<void*>(transferMapping_->colorMapResources_.colorMapTexture));
		mappingView_->draw();
	}
}

auto TransferMappingController::updateModel(Model& model) const -> bool
{
	if (mappingView_->hasNewDataAvailable())
	{
		auto cudaTransferResource = model.transferFunctionGraphicsResource;


		OWL_CUDA_CHECK(cudaGraphicsMapResources(1, &cudaTransferResource));

		cudaArray_t transferFunctionCudaArray{ nullptr };
		OWL_CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&transferFunctionCudaArray, cudaTransferResource, 0, 0));

		const auto resourceDescriptor =
			cudaResourceDesc{ .resType = cudaResourceTypeArray, .res = cudaArray_t{ transferFunctionCudaArray } };


		constexpr auto textureDescriptor =
			cudaTextureDesc{ .addressMode = { cudaAddressModeClamp, cudaAddressModeClamp },
							 .filterMode = cudaFilterModeLinear,
							 .readMode = cudaReadModeElementType,
							 .sRGB = 0,
							 .borderColor = { 1.0f, 1.0f, 1.0f, 1.0f },
							 .normalizedCoords = 1,
							 .maxAnisotropy = 1,
							 .mipmapFilterMode = cudaFilterModePoint,
							 .minMipmapLevelClamp = 0,
							 .maxMipmapLevelClamp = 0 };


		auto transferFunctionCudaTexture = cudaTextureObject_t{};
		OWL_CUDA_CHECK(
			cudaCreateTextureObject(&transferFunctionCudaTexture, &resourceDescriptor, &textureDescriptor, nullptr));


		const auto& dataPoints = mappingView_->resampleData(model.transferFunctionSamplesCount);
		const auto widthBytes = model.transferFunctionSamplesCount * sizeof(float);

		cudaMemcpy2DToArrayAsync(transferFunctionCudaArray, 0, 0, dataPoints.data(), widthBytes, widthBytes, 1,
								 cudaMemcpyHostToDevice);
		OWL_CUDA_CHECK(cudaDestroyTextureObject(transferFunctionCudaTexture));
		OWL_CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaTransferResource));

		model.coloringMode = mappingView_->getColoringMode() == 0 ? b3d::renderer::ColoringMode::single :
																	b3d::renderer::ColoringMode::colormap;

		model.selectedColorMap = mappingView_->getColoringMap();
	}
	return mappingView_->hasNewDataAvailable();
}
