#pragma once

#include "GizmoHelperBase.h"

namespace b3d::renderer
{
	class NullGizmoHelper final : public b3d::renderer::GizmoHelperBase
	{
	public:
		inline auto drawGizmo([[maybe_unused]] owl::AffineSpace3f& transform) -> void override
		{
		}
		inline auto drawBoundGizmo([[maybe_unused]] owl::AffineSpace3f& transform,
								   [[maybe_unused]] const owl::AffineSpace3f& worldTransform,
								   [[maybe_unused]] const owl::vec3f& boxSize) -> void override {};
	};
} // namespace b3d::renderer
