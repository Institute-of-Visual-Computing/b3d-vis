#include "NullRenderer.h"
#include <Logging.h>

using namespace b3d::renderer;

auto NullRenderer::onRender(const View& view) -> void
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
