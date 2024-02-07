#pragma once

#include <owl/common.h>

namespace b3d::renderer
{
	class DebugDrawListBase
	{
	public:
		virtual auto drawBox(const owl::vec3f& midPoint, const owl::vec3f& extent,
							 owl::vec4f color = owl::vec4f{ 1.0f, 1.0f, 1.0f, 1.0f },
							 const owl::LinearSpace3f& orientation = owl::LinearSpace3f{}) -> void = 0;
		virtual ~DebugDrawListBase()
		{
		}
	};
} // namespace b3d::renderer
