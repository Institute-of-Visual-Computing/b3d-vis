#pragma once

#include "DebugDrawListBase.h"

namespace b3d::renderer
{
	class NullDebugDrawList final : public b3d::renderer::DebugDrawListBase
	{
	public:
		auto drawBox([[maybe_unused]] const owl::vec3f& origin, [[maybe_unused]] const owl::vec3f& midPoint,
					 [[maybe_unused]] const owl::vec3f& extent, [[maybe_unused]] owl::vec4f color,
					 [[maybe_unused]] const owl::LinearSpace3f& orientation) -> void override
		{
		}
	};
} // namespace b3d::renderer
