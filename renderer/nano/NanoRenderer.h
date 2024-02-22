#pragma once
#include <RendererBase.h>

#include "nanovdb/util/CreateNanoGrid.h"
#include "owl/owl_host.h"

#include <CudaGpuTimers.h>

#include "features/ColorMapFeature.h"
#include "features/TransferFunctionFeature.h"

namespace b3d::renderer
{
	struct NanoContext
	{
		OWLContext context;
		OWLRayGen rayGen;
		OWLMissProg missProgram;
		OWLGroup worldGeometryGroup;
		OWLLaunchParams launchParams;
	};

	class NanoRenderer final : public RendererBase
	{
	public:
		NanoRenderer()
		{
			colorMapFeature_ = addFeature<ColorMapFeature>("Color Filtering");
			transferFunctionFeature_ = addFeature<TransferFunctionFeature>("Transfer Function");
		}
	protected:
		auto onRender() -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;

	private:
		auto prepareGeometry() -> void;

		NanoContext nanoContext_{};
		owl::AffineSpace3f trs_{};

		nanovdb::Map currentMap_{};

		CudaGpuTimers<100, 4> gpuTimers_{};

		owl::AffineSpace3f renormalizeScale_{};

		ColorMapFeature* colorMapFeature_;
		TransferFunctionFeature* transferFunctionFeature_;
	};
} // namespace b3d::renderer
