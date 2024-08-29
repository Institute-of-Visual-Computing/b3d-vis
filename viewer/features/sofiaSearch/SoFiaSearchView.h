#pragma once

#include "SofiaParams.h"


#include "owl/common/math/AffineSpace.h"

#include "framework/DockableWindowViewBase.h"

class GizmoHelper;

class SoFiaSearchView  final : public DockableWindowViewBase
{
public:
	struct Model
	{
		b3d::tools::sofia::SofiaParams params {};
	};

	SoFiaSearchView(ApplicationContext& appContext, Dockspace* dockspace);
	~SoFiaSearchView() override;
	
	auto setModel(Model model) -> void;
	auto getModel() const -> const Model&;

private:
	auto onDraw() -> void override;

	Model model_;
	owl::AffineSpace3f transform_{};

	float minSNR_{ 3.0f };

	
	std::shared_ptr<GizmoHelper> gizmoHelper_;
};
