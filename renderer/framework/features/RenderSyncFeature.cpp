#include "RenderSyncFeature.h"

#include "Logging.h"

#include <owl/helper/cuda.h>

#include <format>

auto b3d::renderer::RenderSyncFeature::beginUpdate() -> void
{
	synchronization_ = sharedParameters_->get<Synchronization>("synchronization");

	skipUpdate = synchronization_ == nullptr;

	if (skipUpdate)
	{
		b3d::renderer::log("RenderTargetFeature skips update, because of missing shared parameters!");
		return;
	}
	b3d::renderer::log(std::format("wait for fenceVal {}",  synchronization_->fenceValue));
	auto waitParams = cudaExternalSemaphoreWaitParams{};
	waitParams.flags = 0;
	waitParams.params.fence.value = synchronization_->fenceValue;
	OWL_CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&synchronization_->signalSemaphore, &waitParams, 1));
}

auto b3d::renderer::RenderSyncFeature::endUpdate() -> void
{
	if (skipUpdate)
	{
		return;
	}
	b3d::renderer::log(std::format("signal fenceVal {}", synchronization_->fenceValue));

	auto signalParams = cudaExternalSemaphoreSignalParams{};
	signalParams.flags = 0;
	signalParams.params.fence.value = synchronization_->fenceValue;
	OWL_CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&synchronization_->waitSemaphore, &signalParams, 1));
}
