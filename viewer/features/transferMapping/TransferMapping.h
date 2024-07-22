#pragma once
#include "GLUtils.h"

#include <ColorMap.h>
#include <Common.h>

#include "framework/RendererExtensionBase.h"

class TransferMapping : public RendererExtensionBase
{
public:
	TransferMapping(ApplicationContext& applicationContext) : RendererExtensionBase(applicationContext)
	{
		//applicationContext
	}

private:
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
};
