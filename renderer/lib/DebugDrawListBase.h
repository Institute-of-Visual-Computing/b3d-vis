#pragma once

#include <owl/common.h>

namespace b3d::renderer
{
	class DebugDrawListBase
	{
	public:
		virtual auto drawBox(const owl::vec3f& midPoint, const owl::vec3f& extent,
							 owl::vec3f color = owl::vec3f{ 1.0f, 1.0f, 1.0f },
							 const owl::vec3f& origin = owl::vec3f{ 0.0, 0.0, 0.0 },
							 const owl::Quaternion3f& orientation = owl::Quaternion3f{ 1.0, 0.0, 0.0, 0.0 })
			-> void = 0;
		virtual auto drawBox(const owl::vec3f& midPoint, const owl::vec3f& extent,
							 owl::vec3f color = owl::vec3f{ 1.0f, 1.0f, 1.0f },
							 const owl::LinearSpace3f& orientation = owl::LinearSpace3f{})
			-> void = 0;
		virtual ~DebugDrawListBase()
		{
		}
	};
} // namespace b3d::renderer
