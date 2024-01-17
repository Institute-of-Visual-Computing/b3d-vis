#include "GizmoHelper.h"

auto GizmoHelper::drawGizmo(owl::AffineSpace3f& transform) -> void
{
	transforms_.push_back(&transform);
}
