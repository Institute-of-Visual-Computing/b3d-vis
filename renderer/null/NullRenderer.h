#pragma once
#include <RendererBase.h>

namespace b3d::renderer
{
	class NullRenderer final : public RendererBase
	{
	public:
		NullRenderer()
		{
			rendererState_ = std::make_unique<RendererState>();
		}

	protected:
		auto onRender(const View& view) -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;
	};
} // namespace b3d
