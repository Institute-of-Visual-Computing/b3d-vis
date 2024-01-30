#pragma once

#include "DebugDrawListBase.h"

namespace b3d::renderer
{
	class NullDebugDrawList final : public b3d::renderer::DebugDrawListBase
	{
	public:
		auto drawBox(const owl::vec3f& midPoint, const owl::vec3f& extent, owl::vec3f color, const owl::vec3f& origin,
					 const owl::Quaternion3f& orientation) -> void override
		{
		}
		auto drawBox(const owl::vec3f& midPoint, const owl::vec3f& extent, owl::vec3f color,
					 const owl::LinearSpace3f& orientation) -> void override
		{
		}
	};
} // namespace b3d::renderer
