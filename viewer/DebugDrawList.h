#pragma once
#include <vector>

#include "DebugDrawVertex.h"

#include "DebugDrawListBase.h"

class DebugDrawList final : public b3d::renderer::DebugDrawListBase
{
private:
	friend class DebugDrawPass;

	auto reset() -> void;

public:
	// void drawSphere(const glm::vec3& midPoint, float radius, glm::vec4 color = glm::vec4{1.0f,1.0f,1.0f,1.0f});
	// void drawLine(const glm::vec3& p0, const glm::vec3& p1, glm::vec4 color = glm::vec4{1.0f,1.0f,1.0f,1.0f});
	auto drawBox(const owl::vec3f& origin, const owl::vec3f& midPoint, const owl::vec3f& extent, owl::vec4f color,
				 const owl::LinearSpace3f& orientation) -> void override;

private:
	std::vector<DebugDrawVertex> vertices_;
};
