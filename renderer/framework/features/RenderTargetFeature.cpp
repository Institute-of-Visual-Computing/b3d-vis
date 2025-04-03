#include "RenderTargetFeature.h"

#include <format>

#include "Logging.h"
#include <owl/helper/cuda.h>

auto b3d::renderer::RenderTargetFeature::beginUpdate() -> void
{
	static int frmIdx = 0;
	frmIdx++;
	renderTargets_ = sharedParameters_->get<RenderTargets>("renderTargets");
	
	skipUpdate = renderTargets_ == nullptr;

	auto id = std::this_thread::get_id();

	if (skipUpdate)
	{
		b3d::renderer::log("RenderTargetFeature skips update, because of missing shared parameters!");
		return;
	}

	// TODO: minmaxRT not considered yet
	{
		//TODO: Remove dead code
		/*using namespace std::chrono_literals;
		if (frmIdx == 1)
		{
			std::this_thread::sleep_for(0.6s);
		}*/
		
		OWL_CUDA_CHECK(
			cudaGraphicsMapResources(1, &renderTargets_->colorRt.target));
			

		for (auto i = 0u; i < renderTargets_->colorRt.extent.depth; i++)
		{
			OWL_CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cudaColorRT_.surfaces[i].buffer, renderTargets_->colorRt.target, i, 0u));

			auto resDesc = cudaResourceDesc{};
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cudaColorRT_.surfaces[i].buffer;

			OWL_CUDA_CHECK(cudaCreateSurfaceObject(&cudaColorRT_.surfaces[i].surface, &resDesc))
		}
	}
	cudaColorRT_.extent = renderTargets_->colorRt.extent;
}

auto b3d::renderer::RenderTargetFeature::endUpdate() -> void
{
	if (skipUpdate)
	{
		return;
	}

	for (auto i = 0u; i < renderTargets_->colorRt.extent.depth; i++)
	{
		OWL_CUDA_CHECK(cudaDestroySurfaceObject(cudaColorRT_.surfaces[i].surface));
	}
	OWL_CUDA_CHECK(cudaGraphicsUnmapResources(1, &renderTargets_->colorRt.target));
}

auto b3d::renderer::RenderTargetFeature::getParamsData() -> ParamsData
{
	return { .colorRT = cudaColorRT_, .minMaxRT = cudaMinMaxRT_};
}
