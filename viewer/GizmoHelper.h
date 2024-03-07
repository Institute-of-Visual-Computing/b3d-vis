#pragma once

#include <GizmoHelperBase.h>
#include <vector>

class GizmoHelper final : public b3d::renderer::GizmoHelperBase
{
	struct BoundTransform
	{
		owl::vec3f bound{};
		owl::AffineSpace3f* transform{nullptr};
	};
public:
	auto drawGizmo(owl::AffineSpace3f& transform) -> void override;
	inline auto getTransforms() -> std::vector<owl::AffineSpace3f*>
	{
		return transforms_;
	}

	inline auto getBoundTransforms() -> std::vector<BoundTransform>
	{
		return boundTransforms_;
	}

	auto drawBoundGizmo(owl::AffineSpace3f& transform, const owl::vec3f& boxSize) -> void override;

	inline auto clear() -> void
	{
		transforms_.clear();
		boundTransforms_.clear();
	}

private:

	

	std::vector<owl::AffineSpace3f*> transforms_{};
	std::vector<BoundTransform> boundTransforms_{};
};
