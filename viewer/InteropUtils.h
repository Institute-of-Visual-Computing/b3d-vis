#pragma once

#include <glm/glm.hpp>
#include <owl/common/math/vec.h>




[[nodiscard]] inline owl::vec3f owl_cast(const glm::vec3& value)
{
	return owl::vec3f{value.x, value.y, value.z};
}
