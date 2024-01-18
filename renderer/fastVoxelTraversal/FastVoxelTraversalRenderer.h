#pragma once

#include <RendererBase.h>

#include "owl/owl_host.h"

namespace b3d::renderer
{

	struct FastVoxelTraversalNativeRenderingData : RendererState
	{
		float transferOffset{ 0.5f };
	};

	class FastVoxelTraversalRenderer final : public RendererBase
	{

	public:
		FastVoxelTraversalRenderer()
		{
			rendererState_ = std::make_unique<FastVoxelTraversalNativeRenderingData>();
		}

	protected:
		auto onRender(const View& view) -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;

		bool sbtDirty = true;
		owl2i fbSize_{ 0, 0 };

		OWLRayGen rayGen_{ nullptr };
		OWLMissProg missProg_{ nullptr };
		OWLContext context_{ nullptr };
		OWLBuffer surfaceBuffer_{ nullptr };
		OWLLaunchParams launchParameters_{};
		OWLGroup world_{ nullptr };

		OWLGeomType aabbGeomType_{ nullptr };

		OWLGeom concreteAabbGeom_{ nullptr };
		OWLGroup aabbGroup_{ nullptr };

		OWLTexture transferTexture1D_{ nullptr };

		
		float integral_{ 0 };
		float invIntegral_{ 0 };

	};
} // namespace b3d::renderer
