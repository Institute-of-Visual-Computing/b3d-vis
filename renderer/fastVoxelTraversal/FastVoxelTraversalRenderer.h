#pragma once

#include <RendererBase.h>

namespace b3d
{
	struct FastVoxelTraversalRenderer final : public RendererBase
	{
		void onRender(const b3d::View& view) override;
		void onInitialize() override;
		void onDeinitialize() override;
		void onGui() override;
	};
} // namespace b3d
