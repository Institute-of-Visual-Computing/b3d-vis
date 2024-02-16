#include "RendererBase.h"
#include <vector>

using namespace b3d::renderer;

std::vector<RendererRegistryEntry> b3d::renderer::registry;

auto RendererBase::initialize(RenderingDataBuffer* renderData,
							  const DebugInitializationInfo& debugInitializationInfo) -> void
{
	renderData_ = renderData;
	debugInfo_ = debugInitializationInfo;
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

auto RendererBase::render() -> void
{
	onRender();
}

