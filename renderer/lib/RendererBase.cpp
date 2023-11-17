#include "RendererBase.h"
#include <vector>

using namespace b3d::renderer;

std::vector<RendererRegistryEntry> b3d::renderer::registry;

auto RendererBase::initialize() -> void
{
	onInitialize();
}
auto RendererBase::deinitialize() -> void
{
	onDeinitialize();
}
auto RendererBase::gui() -> void
{
	onGui();
}
auto RendererBase::render(const View& view) -> void
{
	onRender(view);
}
