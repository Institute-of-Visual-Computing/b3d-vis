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
			renderSyncFeature_ = addFeature<RenderSyncFeature>("Main Synchronization");
			renderTargetFeature_ = addFeature<RenderTargetFeature>("RenderTargets");
		}
	protected:
		auto onRender() -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;

		RenderSyncFeature* renderSyncFeature_;
		RenderTargetFeature* renderTargetFeature_;
	};
} // namespace b3d::renderer
