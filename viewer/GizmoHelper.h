#pragma once

#include <GizmoHelperBase.h>
#include <vector>

class GizmoHelper final : public b3d::renderer::GizmoHelperBase
{
public:
	auto drawGizmo(owl::AffineSpace3f& transform) -> void override;
	inline auto getTransforms() -> std::vector<owl::AffineSpace3f*>
	{
		return transforms_;
	}

	auto clear(){transforms_.clear();}

private:
	std::vector<owl::AffineSpace3f*> transforms_{};
};
