#include "RendererBase.h"
#include <vector>

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"

using namespace b3d::renderer;

std::vector<RendererRegistryEntry> b3d::renderer::registry;

auto RendererBase::initialize(RenderingDataBuffer* renderData, const DebugInitializationInfo& debugInitializationInfo)
	-> void
{
	renderData_ = renderData;
	debugInfo_ = debugInitializationInfo;

	for (const auto& feature : renderFeatures_)
	{
		feature->initialize(*renderData);
	}

	onInitialize();
}

auto RendererBase::deinitialize() -> void
{
	onDeinitialize();
}

auto RendererBase::gui() -> void
{
	onGui();

	ImGui::Begin("Features");

	for (const auto& feature : renderFeatures_)
	{
		if (ImGui::CollapsingHeader(feature->featureName().c_str()))
		{
			feature->gui();
		}
	}

	ImGui::End();
}

auto RendererBase::render() -> void
{
	for (const auto& feature : renderFeatures_)
	{
		feature->beginUpdate();
	}

	onRender();

	for (const auto& feature : renderFeatures_)
	{
		feature->endUpdate();
	}
}
