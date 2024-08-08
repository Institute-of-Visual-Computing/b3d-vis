#pragma once

#include "ExtensionBase.h"

namespace b3d::renderer
{
	class RenderingDataWrapper;
}


class RendererExtensionBase : public ExtensionBase
{
public:
	explicit RendererExtensionBase(ApplicationContext& applicationContext) : ExtensionBase(applicationContext)
	{
	}

	virtual auto updateRenderingData(b3d::renderer::RenderingDataWrapper& renderingData) -> void = 0;
};
