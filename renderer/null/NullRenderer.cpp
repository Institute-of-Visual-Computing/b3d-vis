#include "NullRenderer.h"
#include <Logging.h>
using namespace b3d;

auto b3d::NullRenderer::onRender(const b3d::View& view) -> void
{
	log("[NullRenderer] onRender!");
}

auto b3d::NullRenderer::onInitialize() -> void
{
	log("[NullRenderer] onInitialize!");
}

auto b3d::NullRenderer::onDeinitialize() -> void
{
	log("[NullRenderer] onDeinitialize!");
}

auto b3d::NullRenderer::onGui() -> void
{
	log("[NullRenderer] onGui!");
}
