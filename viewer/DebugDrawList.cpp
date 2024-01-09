#include "DebugDrawList.h"

#include <array>

#include "glm/glm.hpp"
#include "glm/gtx/quaternion.hpp"

#include "glm/packing.hpp"

auto DebugDrawList::drawBox(const owl::vec3f& midPoint, const owl::vec3f& extent, owl::vec3f color,
							const owl::Quaternion3f& orientation) -> void
{
	const auto glmMidPoint = glm::vec3{ midPoint.x, midPoint.y, midPoint.z };

	const auto translate = glm::translate(glm::identity<glm::mat4>(), glmMidPoint);
	const auto invTranslate = glm::translate(glm::identity<glm::mat4>(), -glmMidPoint);
	const auto rotate = glm::toMat4(glm::quat{ orientation.r, orientation.i, orientation.j, orientation.k });
	const auto rotor = translate * rotate * invTranslate;

	const auto glmExtent = glm::vec3{ extent.x, extent.y, extent.z };

	/*
	 *	  p4----p5
	 *	 / |   / |
	 *	p0----p1 |
	 *	|  p6--|-p7
	 *	| /    |/
	 *	p2----p3
	 */
	const auto p0 = glm::vec3(rotor * glm::vec4(glmMidPoint + 0.5f * glm::vec3(-1.0, -1.0, 1.0) * glmExtent, 1.0));
	const auto p1 = glm::vec3(rotor * glm::vec4(glmMidPoint + 0.5f * glm::vec3(1.0, -1.0, 1.0) * glmExtent, 1.0f));
	const auto p2 = glm::vec3(rotor * glm::vec4(glmMidPoint + 0.5f * glm::vec3(-1.0, -1.0, -1.0) * glmExtent, 1.0f));
	const auto p3 = glm::vec3(rotor * glm::vec4(glmMidPoint + 0.5f * glm::vec3(1.0, -1.0, -1.0) * glmExtent, 1.0f));
	const auto p4 = glm::vec3(rotor * glm::vec4(glmMidPoint + 0.5f * glm::vec3(-1.0, 1.0, 1.0) * glmExtent, 1.0f));
	const auto p5 = glm::vec3(rotor * glm::vec4(glmMidPoint + 0.5f * glm::vec3(1.0, 1.0, 1.0) * glmExtent, 1.0f));
	const auto p6 = glm::vec3(rotor * glm::vec4(glmMidPoint + 0.5f * glm::vec3(-1.0, 1.0, -1.0) * glmExtent, 1.0f));
	const auto p7 = glm::vec3(rotor * glm::vec4(glmMidPoint + 0.5f * glm::vec3(1.0, 1.0, -1.0) * glmExtent, 1.0f));

	const auto packedColor = glm::packUnorm4x8(glm::vec4(color.x, color.y, color.z, 1.0f));

	auto triangles = std::array<DebugDrawVertex, 72>{};

	auto index = 0;
	// p0->p1->p2
	triangles[index++] = DebugDrawVertex{ p0, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p1, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p2, glm::vec2{ 0.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p0, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p1, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p2, glm::vec2{ 1.0f, 0.0f }, packedColor };
	// p1->p3->p2
	triangles[index++] = DebugDrawVertex{ p1, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p3, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p2, glm::vec2{ 1.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p1, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p3, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p2, glm::vec2{ 0.0f, 0.0f }, packedColor };
	// p1->p5->p3
	triangles[index++] = DebugDrawVertex{ p1, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p5, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p3, glm::vec2{ 1.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p1, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p5, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p3, glm::vec2{ 0.0f, 0.0f }, packedColor };
	// p5->p7->p3
	triangles[index++] = DebugDrawVertex{ p5, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p7, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p3, glm::vec2{ 1.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p5, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p7, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p3, glm::vec2{ 0.0f, 0.0f }, packedColor };
	// p2->p3->p6
	triangles[index++] = DebugDrawVertex{ p2, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p3, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 1.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p2, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p3, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 0.0f, 0.0f }, packedColor };
	// p3->p7->p6
	triangles[index++] = DebugDrawVertex{ p3, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p7, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 1.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p3, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p7, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 0.0f, 0.0f }, packedColor };
	// p0->p2->p6
	triangles[index++] = DebugDrawVertex{ p0, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p2, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 1.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p0, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p2, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 0.0f, 0.0f }, packedColor };
	// p0->p6->p4
	triangles[index++] = DebugDrawVertex{ p0, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p4, glm::vec2{ 0.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p0, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p4, glm::vec2{ 0.0f, 0.0f }, packedColor };
	// p0->p4->p1
	triangles[index++] = DebugDrawVertex{ p0, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p4, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p1, glm::vec2{ 0.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p0, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p4, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p1, glm::vec2{ 1.0f, 0.0f }, packedColor };
	// p1->p4->p5
	triangles[index++] = DebugDrawVertex{ p4, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p5, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p1, glm::vec2{ 0.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p4, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p5, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p1, glm::vec2{ 1.0f, 0.0f }, packedColor };
	// p4->p6->p5
	triangles[index++] = DebugDrawVertex{ p4, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p5, glm::vec2{ 0.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p4, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p5, glm::vec2{ 1.0f, 0.0f }, packedColor };
	// p6->p7->p5
	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 1.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p7, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p5, glm::vec2{ 0.0f, 0.0f }, packedColor };

	triangles[index++] = DebugDrawVertex{ p6, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p7, glm::vec2{ 0.0f, 0.0f }, packedColor };
	triangles[index++] = DebugDrawVertex{ p5, glm::vec2{ 1.0f, 0.0f }, packedColor };

	vertices_.insert(vertices_.end(), triangles.begin(), triangles.end());
}
auto DebugDrawList::reset() -> void
{
	vertices_.clear();
}
