#include "DebugDrawList.h"

#include <array>

#include "glm/glm.hpp"
#include "glm/gtx/quaternion.hpp"

#include "glm/packing.hpp"

auto DebugDrawList::drawBox(const owl::vec3f& midPoint, const owl::vec3f& extent, owl::vec3f color,
							const owl::vec3f& origin, const owl::Quaternion3f& orientation) -> void
{
	const auto glmMidPoint = glm::vec3{ midPoint.x, midPoint.y, midPoint.z };
	const auto glmOrigin = glm::vec3{ origin.x, origin.y, origin.z };

	const auto translate = glm::translate(glm::identity<glm::mat4>(), glmOrigin);
	const auto invTranslate = glm::translate(glm::identity<glm::mat4>(), -glmOrigin);
	const auto rotate = glm::toMat4(glm::quat{ orientation.r, orientation.i, orientation.j, orientation.k });
	const auto de = glm::determinant(rotate);
	std::cout << "§§§§§§§§§§§§§§      " << std::to_string(de) << std::endl;
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

auto DebugDrawList::drawBox(const owl::vec3f& midPoint, const owl::vec3f& extent, owl::vec3f color,
							const owl::LinearSpace3f& orientation) -> void
{

	float mat[16];

	mat[3] = 0.0f;
	mat[7] = 0.0f;
	mat[11] = 0.0f;

	mat[12] = midPoint.x;
	mat[13] = midPoint.y;
	mat[14] = midPoint.z;

	mat[15] = 1.0f;



	mat[0] = orientation.vx.x;
	mat[1] = orientation.vx.y;
	mat[2] = orientation.vx.z;

	mat[4] = orientation.vy.x;
	mat[5] = orientation.vy.y;
	mat[6] = orientation.vy.z;

	mat[8] = orientation.vz.x;
	mat[9] = orientation.vz.y;
	mat[10] = orientation.vz.z;

	auto transform = glm::mat4{
		mat[0],
		mat[1],
		mat[2],
		mat[3],
		mat[4],
		mat[5],
		mat[6],
		mat[7],
		mat[8],
		mat[9],
		mat[10],
		mat[11],
		mat[12],
		mat[13],
		mat[14],
		mat[15]
	};

	const auto glmMidPoint = glm::vec3{ midPoint.x, midPoint.y, midPoint.z };


	const auto glmExtent = glm::vec3{ extent.x, extent.y, extent.z };

	/*
	 *	  p4----p5
	 *	 / |   / |
	 *	p0----p1 |
	 *	|  p6--|-p7
	 *	| /    |/
	 *	p2----p3
	 */
	const auto p0 = glm::vec3(transform * glm::vec4(glmMidPoint + 0.5f * glm::vec3(-1.0, -1.0, 1.0) * glmExtent, 1.0));
	const auto p1 = glm::vec3(transform * glm::vec4(glmMidPoint + 0.5f * glm::vec3(1.0, -1.0, 1.0) * glmExtent, 1.0f));
	const auto p2 = glm::vec3(transform * glm::vec4(glmMidPoint + 0.5f * glm::vec3(-1.0, -1.0, -1.0) * glmExtent, 1.0f));
	const auto p3 = glm::vec3(transform * glm::vec4(glmMidPoint + 0.5f * glm::vec3(1.0, -1.0, -1.0) * glmExtent, 1.0f));
	const auto p4 = glm::vec3(transform * glm::vec4(glmMidPoint + 0.5f * glm::vec3(-1.0, 1.0, 1.0) * glmExtent, 1.0f));
	const auto p5 = glm::vec3(transform * glm::vec4(glmMidPoint + 0.5f * glm::vec3(1.0, 1.0, 1.0) * glmExtent, 1.0f));
	const auto p6 = glm::vec3(transform * glm::vec4(glmMidPoint + 0.5f * glm::vec3(-1.0, 1.0, -1.0) * glmExtent, 1.0f));
	const auto p7 = glm::vec3(transform * glm::vec4(glmMidPoint + 0.5f * glm::vec3(1.0, 1.0, -1.0) * glmExtent, 1.0f));

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
