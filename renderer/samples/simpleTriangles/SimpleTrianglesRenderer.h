#pragma once
#include "ColorMap.h"
#include "RendererBase.h"
#include "owl/owl_host.h"

class SimpleTrianglesRenderer final : public b3d::renderer::RendererBase
{
public:
	SimpleTrianglesRenderer() 
	{
	}

	auto onGui() -> void override;
protected:
	auto onRender() -> void override;
	auto onInitialize() -> void override;

	bool sbtDirty = true;
	owl2i fbSize_ { 0, 0 };

	OWLRayGen rayGen_{ nullptr };
	OWLMissProg missProg_{ nullptr };
	OWLContext context_{ nullptr };
	OWLBuffer surfaceBuffer_{ nullptr };
	OWLParams launchParameters_{ };
	OWLGroup world_{ nullptr };

	OWLTexture colorMapTexture_;

	b3d::tools::colormap::ColorMap colorMap_;

};
