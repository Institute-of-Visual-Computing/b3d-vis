#include "NullRenderer.h"
#include <Logging.h>
#include "cuda_runtime.h"

using namespace b3d::renderer;


auto NullRenderer::onRender() -> void
{
	log("[NullRenderer] onRender!");

	const auto synchronization = renderData_->get<Synchronization>("synchronization");
	
	auto waitParams = cudaExternalSemaphoreWaitParams{};
	waitParams.flags = 0;
	waitParams.params.fence.value = synchronization->fenceValue;
	cudaWaitExternalSemaphoresAsync(&synchronization->signalSemaphore, &waitParams, 1);

	auto signalParams = cudaExternalSemaphoreSignalParams{};
	signalParams.flags = 0;
	signalParams.params.fence.value = synchronization->fenceValue;
	cudaSignalExternalSemaphoresAsync(&synchronization->waitSemaphore, &signalParams, 1);
}

auto NullRenderer::onInitialize() -> void
{
	log("[NullRenderer] onInitialize!");
}

auto NullRenderer::onDeinitialize() -> void
{
	log("[NullRenderer] onDeinitialize!");
}

auto NullRenderer::onGui() -> void
{
	log("[NullRenderer] onGui!");
}
