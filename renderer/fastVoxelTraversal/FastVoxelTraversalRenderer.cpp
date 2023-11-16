#include "FastVoxelTraversalRenderer.h"
#include "Logging.h"

using namespace b3d::renderer;

auto FastVoxelTraversalRenderer::onRender(const View& view) -> void
{
}

auto FastVoxelTraversalRenderer::onInitialize() -> void
{
	log("[FastVoxelTraversalRenderer] onInitialize!");
}

auto FastVoxelTraversalRenderer::onDeinitialize() -> void
{
	log("[FastVoxelTraversalRenderer] onDeinitialize!");
}

auto FastVoxelTraversalRenderer::onGui() -> void
{
	log("[FastVoxelTraversalRenderer] onGui!");
}
