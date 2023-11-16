#include "NullRenderer.h"
#include <Logging.h>
using namespace b3d;

void b3d::NullRenderer::onRender(const b3d::View& view)
{
	log("[NullRenderer] onRender!");
}

void b3d::NullRenderer::onInitialize()
{
	log("[NullRenderer] onInitialize!");
}

void b3d::NullRenderer::onDeinitialize()
{
	log("[NullRenderer] onDeinitialize!");
}

void b3d::NullRenderer::onGui()
{
	log("[NullRenderer] onGui!");
}
