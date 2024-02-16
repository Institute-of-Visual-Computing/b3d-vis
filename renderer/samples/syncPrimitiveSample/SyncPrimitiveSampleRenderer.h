#pragma once
#include <RendererBase.h>

namespace b3d::renderer
{
	class SyncPrimitiveSampleRenderer final : public RendererBase
	{
	public:
		SyncPrimitiveSampleRenderer()
		{
		}
	protected:
		auto onRender() -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;
	};
} // namespace b3d::renderer
