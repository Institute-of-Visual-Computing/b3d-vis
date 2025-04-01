#include "NullRenderer.h"
#include <Logging.h>
#include <cuda_runtime.h>

using namespace b3d::renderer;


auto NullRenderer::onRender() -> void
{
	log("[NullRenderer] onRender!");
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
