#pragma once

#define IMGUI_DEFINE_MATH_OPERATORS
#include <RendererBase.h>

#include "nanovdb/util/CreateNanoGrid.h"
#include "owl/owl_host.h"

#include <CudaGpuTimers.h>


#include "NanoCutterParser.h"
#include "OpenFileDialog.h"
#include "features/RenderSyncFeature.h"
#include "features/RenderTargetFeature.h"


namespace b3d::renderer::nano
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
			renderTargetFeature_ = addFeature<RenderTargetFeature>("RenderTargets");
			// renderSyncFeature_ = addFeatureWithDependency<RenderSyncFeature>({ renderTargetFeature_ }, "Main Synchronization");
		}

	protected:
		auto onRender() -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto loadDataSet(const cutterParser::B3DDataSet& dataSet) -> void;
		auto onGui() -> void override;

	private:
		auto prepareGeometry() -> void;

		NanoContext nanoContext_{};
		owl::AffineSpace3f trs_{};

		nanovdb::Map currentMap_{};

		CudaGpuTimers<100, 4> gpuTimers_{};

		OpenFileDialog openFileDialog_{ SelectMode::singleFile, { ".b3d" } };

		std::optional<cutterParser::B3DDataSet> dataSet_{ std::nullopt };
		std::array<int, 2> visibleLevelRange{ 0, 10 };


		RenderTargetFeature* renderTargetFeature_;
		// RenderSyncFeature* renderSyncFeature_;
	};
} // namespace b3d::renderer::nano
