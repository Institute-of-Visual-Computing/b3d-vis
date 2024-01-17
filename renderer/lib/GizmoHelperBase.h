#pragma once

#include <owl/common.h>

namespace b3d::renderer
{
	class GizmoHelperBase
	{
	public:
		virtual auto drawGizmo(owl::AffineSpace3f& transform) -> void = 0;
		virtual ~GizmoHelperBase()
		{
		}
	};
} // namespace b3d::renderer
