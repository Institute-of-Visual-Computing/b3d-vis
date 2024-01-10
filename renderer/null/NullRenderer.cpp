#include "NullRenderer.h"
#include <Logging.h>
#include "cuda_runtime.h"

using namespace b3d::renderer;


auto NullRenderer::onRender(const View& view) -> void
{
	log("[NullRenderer] onRender!");
	auto waitParams = cudaExternalSemaphoreWaitParams{ };
	waitParams.flags = 0;
    waitParams.params.fence.value = 1;
	cudaWaitExternalSemaphoresAsync(&initializationInfo_.signalSemaphore, &waitParams, 1);

	constexpr std::array signalParams = { cudaExternalSemaphoreSignalParams{ { { 1 } }, 0 },
										  cudaExternalSemaphoreSignalParams{ { { 0 } }, 0 } };
	cudaSignalExternalSemaphoresAsync(&initializationInfo_.waitSemaphore, signalParams.data(), 2);
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
