#pragma once
#include <RendererBase.h>

#include <features/RenderSyncFeature.h>
#include <features/RenderTargetFeature.h>

namespace b3d::renderer
{
	class TextureFillerSyncSample final : public RendererBase
	{
	public:
		TextureFillerSyncSample()
		{
			renderTargetFeature_ = addFeature<RenderTargetFeature>("RenderTargets");
			// renderSyncFeature_ =addFeatureWithDependency<RenderSyncFeature>({ renderTargetFeature_ }, "Main Synchronization");
		}

	protected:
		auto onRender() -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;

		RenderTargetFeature* renderTargetFeature_;
		// RenderSyncFeature* renderSyncFeature_;
	};
} // namespace b3d::renderer
