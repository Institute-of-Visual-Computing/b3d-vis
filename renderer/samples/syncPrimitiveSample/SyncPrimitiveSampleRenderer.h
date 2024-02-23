#pragma once
#include <RendererBase.h>

#include "features/RenderSyncFeature.h"
#include "features/RenderTargetFeature.h"

namespace b3d::renderer
{
	class SyncPrimitiveSampleRenderer final : public RendererBase
	{
	public:
		SyncPrimitiveSampleRenderer()
		{
			renderSyncFeature_ = addFeature<RenderSyncFeature>("Main Synchronization");
			renderTargetFeature_ = addFeature<RenderTargetFeature>("RenderTargets");
		}
	protected:
		auto onRender() -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;

		RenderTargetFeature* renderTargetFeature_;
		RenderSyncFeature* renderSyncFeature_;
	};
} // namespace b3d::renderer
