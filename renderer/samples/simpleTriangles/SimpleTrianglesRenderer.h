#pragma once
#include "ColorMap.h"
#include "RendererBase.h"
#include "owl/owl_host.h"

#include "features/BackgroundColorFeature.h"
#include "features/ColorMapFeature.h"
#include "features/RenderSyncFeature.h"
#include "features/RenderTargetFeature.h"
#include "features/TransferFunctionFeature.h"

namespace b3d::renderer
{
	class SimpleTrianglesRenderer final : public b3d::renderer::RendererBase
	{
	public:
		SimpleTrianglesRenderer()
		{
			renderTargetFeature_ = addFeature<RenderTargetFeature>("RenderTargets");
			colorMapFeature_ = addFeature<ColorMapFeature>("Color Filtering");
			//transferFunctionFeature_ = addFeature<TransferFunctionFeature>("Transfer Function");
			backgroundColorFeature_ = addFeature<BackgroundColorFeature>(
				"Background Color", std::array<ColorRGB, 2>{ { { 0.572f, 0.100f, 0.750f }, { 0.0f, 0.3f, 0.3f } } });
			renderSyncFeature_ = addFeatureWithDependency<RenderSyncFeature>(
				{ renderTargetFeature_, colorMapFeature_,/* transferFunctionFeature_,*/ backgroundColorFeature_ },
				"Main Synchronization");
		}

		auto onGui() -> void override;

	protected:
		auto onRender() -> void override;
		auto onInitialize() -> void override;

		bool sbtDirty = true;
		owl2i fbSize_{ 0, 0 };

		OWLRayGen rayGen_{ nullptr };
		OWLMissProg missProg_{ nullptr };
		OWLContext context_{ nullptr };
		OWLBuffer surfaceBuffer_{ nullptr };
		OWLParams launchParameters_{};
		OWLGroup world_{ nullptr };

		RenderTargetFeature* renderTargetFeature_;
		RenderSyncFeature* renderSyncFeature_;
		ColorMapFeature* colorMapFeature_;
		//TransferFunctionFeature* transferFunctionFeature_;
		BackgroundColorFeature* backgroundColorFeature_;
	};
} // namespace b3d::renderer
