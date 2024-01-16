#pragma once

#include "DebugDrawListBase.h"

namespace b3d::renderer
{
	class NullDebugDrawList final : public b3d::renderer::DebugDrawListBase
	{
	public:
		inline auto drawBox(const owl::vec3f& midPoint, const owl::vec3f& extent,
							owl::vec3f color = owl::vec3f{ 1.0f, 1.0f, 1.0f },
							const owl::Quaternion3f& orientation = owl::Quaternion3f{ 1.0, 0.0, 0.0, 0.0 })
			-> void override{}
	};
} // namespace b3d::renderer
