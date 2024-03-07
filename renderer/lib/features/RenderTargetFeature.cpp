#include "RenderTargetFeature.h"

#include "Logging.h"
#include "owl/helper/cuda.h"

auto b3d::renderer::RenderTargetFeature::beginUpdate() -> void
{
	renderTargets_ = sharedParameters_->get<RenderTargets>("renderTargets");

	skipUpdate = renderTargets_ == nullptr;

	if (skipUpdate)
	{
		b3d::renderer::log("RenderTargetFeature skips update, because of missing shared parameters!");
		return;
	}

	// TODO: minmaxRT not considered yet
	std::array<cudaArray_t, 2> cudaArrays{};
	{
		OWL_CUDA_CHECK(
			cudaGraphicsMapResources(1, &renderTargets_->colorRt.target));

		for (auto i = 0; i < renderTargets_->colorRt.extent.depth; i++)
		{
			OWL_CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cudaArrays[i], renderTargets_->colorRt.target, i, 0));

			auto resDesc = cudaResourceDesc{};
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cudaArrays[i];

			OWL_CUDA_CHECK(cudaCreateSurfaceObject(&cudaColorRT_.surfaces[i], &resDesc))
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

	for (auto i = 0; i < renderTargets_->colorRt.extent.depth; i++)
	{
		OWL_CUDA_CHECK(cudaDestroySurfaceObject(cudaColorRT_.surfaces[i]));
	}
	OWL_CUDA_CHECK(cudaGraphicsUnmapResources(1, &renderTargets_->colorRt.target));
}

auto b3d::renderer::RenderTargetFeature::getParamsData() -> ParamsData
{
	return { .colorRT = cudaColorRT_, .minMaxRT = cudaMinMaxRT_};
}
