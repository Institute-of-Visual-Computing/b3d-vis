#pragma once
#include "RendererBase.h"
#include "owl/owl_host.h"

using namespace b3d::renderer;

struct NativeCube
{
	owl::vec3f position;
	owl::vec3f scale;
	owl::Quaternion3f rotation;
};
class SimpleTrianglesRenderer final : public RendererBase
{
public:
	auto setCubeVolumeTransform(NativeCube *nt)-> void;
	auto onGui() -> void override;
protected:
	auto onRender(const View& view) -> void override;
	auto onInitialize() -> void override;

	owl::AffineSpace3f trs;

	bool sbtDirty = true;
	OWLRayGen rayGen{ 0 };
	OWLContext context{ 0 };
	OWLBuffer surfaceBuffer_{ nullptr };
	OWLParams launchParameters_{ nullptr };
	OWLGroup world_{ nullptr };
};
