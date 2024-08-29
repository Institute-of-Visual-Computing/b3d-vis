#pragma once

#include <memory>

#include "framework/RendererExtensionBase.h"
#include "framework/UpdatableComponentBase.h"
#include "owl/common/math/AffineSpace.h"

class GizmoHelper;
class SoFiaSearchView;

class SoFiaSearch final : public UpdatableComponentBase, public RendererExtensionBase
{
public:
	SoFiaSearch(ApplicationContext& applicationContext);

	~SoFiaSearch() override;

	auto update() -> void override;
	

private:
	auto initializeResources() -> void override;
	auto deinitializeResources() -> void override;
	auto updateRenderingData(b3d::renderer::RenderingDataWrapper& renderingData) -> void override;

	owl::AffineSpace3f transform_ {};

	bool showSearchWindow_{ true };

	std::unique_ptr<SoFiaSearchView> sofiaSearchView_;
	std::shared_ptr<GizmoHelper> gizmoHelper_;
};
