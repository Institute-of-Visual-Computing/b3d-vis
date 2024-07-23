#pragma once
#include "GLUtils.h"

#include <ColorMap.h>
#include <Common.h>

#include "framework/RendererExtensionBase.h"
#include "TransferMappingController.h"

class TransferMapping final : public RendererExtensionBase
{
public:
	TransferMapping(ApplicationContext& applicationContext);

private:
	constexpr static int transferFunctionSamples = 512;

	friend TransferMappingController;
	struct ColorMapResources
	{
		b3d::tools::colormap::ColorMap colorMap{};
		GLuint colorMapTexture{};
		b3d::renderer::Extent colorMapTextureExtent{};
		cudaGraphicsResource_t cudaGraphicsResource{};
	} colorMapResources_{};

	struct TransferFunctionResources
	{
		GLuint transferFunctionTexture{};
		cudaGraphicsResource_t cudaGraphicsResource{};
	} transferFunctionResources_{};


public:
	auto initializeResources() -> void override;
	auto deinitializeResources() -> void override;
	auto updateRenderingData(b3d::renderer::RenderingDataWrapper& renderingData) -> void override;

	private:
	std::unique_ptr<TransferMappingController> transferMappingController_{};
};
