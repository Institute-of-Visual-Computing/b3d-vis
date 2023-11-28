#include "NanoRenderer.h"
#include <Logging.h>
#include "cuda_runtime.h"

using namespace b3d::renderer;


auto NanoRenderer::onRender(const View& view) -> void
{
	
	auto waitParams = cudaExternalSemaphoreWaitParams{ };
	waitParams.flags = 0;
    waitParams.params.fence.value = 0;
	cudaWaitExternalSemaphoresAsync(&initializationInfo_.signalSemaphore, &waitParams, 1);
	
	auto signalParams = cudaExternalSemaphoreSignalParams{};
	signalParams.flags = 0;
    signalParams.params.fence.value = 0;
	cudaSignalExternalSemaphoresAsync(&initializationInfo_.waitSemaphore, &signalParams, 1);
}

auto NanoRenderer::onInitialize() -> void
{
	log("[NanoRenderer] onInitialize!");
}

auto NanoRenderer::onDeinitialize() -> void
{
	log("[NanoRenderer] onDeinitialize!");
}

auto NanoRenderer::onGui() -> void
{
	
}
