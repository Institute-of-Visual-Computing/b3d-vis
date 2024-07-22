#pragma once

#include "ExtensionBase.h"

namespace b3d::renderer
{
	class RenderingDataWrapper;
} // namespace b3d::renderer


class RendererExtensionBase : public ExtensionBase
{
public:
	RendererExtensionBase(ApplicationContext& applicationContext) : ExtensionBase(applicationContext)
	{
	}

public:
	virtual auto updateRenderingData(b3d::renderer::RenderingDataWrapper& renderingData) -> void = 0;
};
