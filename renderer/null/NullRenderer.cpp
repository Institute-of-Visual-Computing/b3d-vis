#include "NullRenderer.h"
#include <Logging.h>
#include "cuda_runtime.h"

using namespace b3d::renderer;


auto NullRenderer::onRender(const View& view) -> void
{
	log("[NullRenderer] onRender!");

	auto waitParams = cudaExternalSemaphoreWaitParams{};
	waitParams.flags = 0;
	waitParams.params.fence.value = view.fenceValue;
	cudaWaitExternalSemaphoresAsync(&initializationInfo_.signalSemaphore, &waitParams, 1);

	auto signalParams = cudaExternalSemaphoreSignalParams{};
	signalParams.flags = 0;
	signalParams.params.fence.value = view.fenceValue;
	cudaSignalExternalSemaphoresAsync(&initializationInfo_.waitSemaphore, &signalParams, 1);
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
