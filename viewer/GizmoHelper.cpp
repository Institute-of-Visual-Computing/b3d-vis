#include "GizmoHelper.h"

auto GizmoHelper::drawGizmo(owl::AffineSpace3f& transform) -> void
{
	transforms_.push_back(&transform);
}
auto GizmoHelper::drawBoundGizmo(owl::AffineSpace3f& transform, const owl::AffineSpace3f& worldTransform, const owl::vec3f& boxSize) -> void
{
	boundTransforms_.push_back({boxSize, &transform, worldTransform});
}

