#pragma once
#include "RendererBase.h"
#include "owl/owl_host.h"

using namespace b3d::renderer;
class SimpleTrianglesRenderer final : public RendererBase
{
protected:
	auto onRender(const View& view) -> void override;
	auto onInitialize() -> void override;

	bool sbtDirty = true;
	OWLRayGen rayGen{ 0 };
	OWLContext context{ 0 };
	OWLBuffer surfaceBuffer_{ nullptr };
	OWLParams launchParameters_{ nullptr };
	OWLGroup world_{ nullptr };
};
