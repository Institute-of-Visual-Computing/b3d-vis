#pragma once

#include "RendererBase.h"

namespace b3d::renderer
{
	class CudaSurfaceObjectWriteTestRenderer final : public RendererBase
	{
	public:
		CudaSurfaceObjectWriteTestRenderer()
		{
		}
	protected:
		auto onRender() -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;
	};
} // namespace b3d::renderer
