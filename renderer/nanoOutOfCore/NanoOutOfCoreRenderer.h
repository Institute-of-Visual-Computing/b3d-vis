#pragma once
#include <RendererBase.h>

#include "nanovdb/util/CreateNanoGrid.h"
#include "owl/owl_host.h"

#include <CudaGpuTimers.h>


#include "OpenFileDialog.h"


namespace b3d::renderer::nano
{
	struct NanoNativeRenderingData : b3d::renderer::RendererState
	{
		b3d::renderer::VolumeTransform volumeTransform;
	};

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
			rendererState_ = std::make_unique<nano::NanoNativeRenderingData>();
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

		nanovdb::Map currentMap_{};

		CudaGpuTimers<100, 4> gpuTimers_{};

		OpenFileDialog openFileDialog_{ SelectMode::singleFile, {".b3d"}};
	};
} // namespace b3d::renderer::nano
