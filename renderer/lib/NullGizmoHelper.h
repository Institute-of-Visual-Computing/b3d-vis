#pragma once

#include "GizmoHelperBase.h"

namespace b3d::renderer
{
	class NullGizmoHelper final : public b3d::renderer::GizmoHelperBase
	{
	public:
		inline auto drawGizmo(owl::AffineSpace3f& transform)  -> void override{}
	};
} // namespace b3d::renderer
