#pragma once
#include <RendererBase.h>

namespace b3d
{
	class NullRenderer final : RendererBase
	{
		auto onRender(const b3d::View& view) -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;
	};
} // namespace b3d
