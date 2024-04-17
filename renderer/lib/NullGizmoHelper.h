#pragma once

#include "GizmoHelperBase.h"

namespace b3d::renderer
{
	class NullGizmoHelper final : public b3d::renderer::GizmoHelperBase
	{
	public:
		inline auto drawGizmo(owl::AffineSpace3f& transform)  -> void override{}
		inline auto drawBoundGizmo(owl::AffineSpace3f& transform, const owl::AffineSpace3f& worldTransform, const owl::vec3f& boxSize)
			-> void override{};
	};
} // namespace b3d::renderer
