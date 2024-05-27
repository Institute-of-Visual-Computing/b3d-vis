#pragma once

#include "RendererBase.h"

#include "features/RenderSyncFeature.h"
#include "features/RenderTargetFeature.h"

namespace b3d::renderer
{
	class CudaSurfaceObjectWriteTestRenderer final : public RendererBase
	{
	public:
		CudaSurfaceObjectWriteTestRenderer()
		{
			renderTargetFeature_ = addFeature<RenderTargetFeature>("RenderTargets");
			// renderSyncFeature_ = addFeatureWithDependency<RenderSyncFeature>({ renderTargetFeature_ }, "Main Synchronization");
		}

	protected:
		auto onRender() -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;

		// RenderSyncFeature* renderSyncFeature_;
		RenderTargetFeature* renderTargetFeature_;
	};
} // namespace b3d::renderer
