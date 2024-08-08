
#include "TransferMappingController.h"
#include <cuda_runtime.h>
#include <owl/helper/cuda.h>
#include "TransferMapping.h"
#include "TransferMappingView.h"
#include "framework/ApplicationContext.h"
#include "framework/Dockspace.h"

TransferMappingController::TransferMappingController(ApplicationContext& applicationContext,
													 TransferMapping& transferMapping)
	: UpdatableComponentBase(applicationContext),
	  mappingView_{ std::make_unique<TransferMappingView>(applicationContext, applicationContext.getMainDockspace()) },
	  transferMapping_{ &transferMapping }
{
	applicationContext.addMenuToggleAction(
		showToolWindow_, [&](bool isOn) { isOn ? mappingView_->open() : mappingView_->close(); }, "Tools",
		"Transfer Mapping");
}

auto TransferMappingController::update() -> void
{
	if (showToolWindow_)
	{
		mappingView_->setColorMapInfos(transferMapping_->colorMapResources_.colorMap.colorMapNames,
									   (void*)transferMapping_->colorMapResources_.colorMapTexture);
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
		// Create texture
		auto resDesc = cudaResourceDesc{};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = transferFunctionCudaArray;


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

		cudaTextureObject_t transferFunctionCudaTexture{};
		OWL_CUDA_CHECK(cudaCreateTextureObject(&transferFunctionCudaTexture, &resDesc, &texDesc, nullptr));


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
