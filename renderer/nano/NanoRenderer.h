#pragma once
#include <RendererBase.h>

#include "nanovdb/util/CreateNanoGrid.h"

struct NanoNativeRenderingData : b3d::renderer::RendererState
{
	b3d::renderer::VolumeTransform volumeTransform;
};

namespace b3d::renderer
{
	struct NanoContext
	{
		OWLContext context;
		OWLRayGen rayGen;
		OWLLaunchParams lp;
		OWLMissProg missProgram;
		OWLGroup worldGeometryGroup;
	};

	class NanoRenderer final : public RendererBase
	{
	public:
		NanoRenderer()
		{
			rendererState_ = std::make_unique<NanoNativeRenderingData>();
		}
	protected:
		auto onRender(const View& view) -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;

	private:
		auto prepareGeometry() -> void;

		NanoContext nanoContext_{};
		owl::AffineSpace3f trs_{};

		nanovdb::Map currentMap{};
	};
} // namespace b3d::renderer
