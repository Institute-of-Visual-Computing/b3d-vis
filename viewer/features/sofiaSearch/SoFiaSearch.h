#pragma once

#include <memory>

#include "SofiaParams.h"
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

	auto initializeResources() -> void override;
	auto deinitializeResources() -> void override;

private:
	auto updateRenderingData(b3d::renderer::RenderingDataWrapper& renderingData) -> void override;

	bool showSearchWindow_{ true };

	auto startSearch(b3d::tools::sofia::SofiaParams params) -> void;

	std::unique_ptr<SoFiaSearchView> sofiaSearchView_;
};
