#pragma once
#include <vector>

#include "DebugDrawVertex.h"

#include "DebugDrawListBase.h"

class DebugDrawList final : public b3d::renderer::DebugDrawListBase
{
public:
	// void drawAABBox(const glm::vec3& lower, const glm::vec3& upper, glm::vec4 color =
	// glm::vec4{1.0f,1.0f,1.0f,1.0f});
	auto drawBox(const owl::vec3f& midPoint, const owl::vec3f& extent, owl::vec3f color, const owl::vec3f& origin,
				 const owl::Quaternion3f& orientation) -> void override;
	// void drawSphere(const glm::vec3& midPoint, float radius, glm::vec4 color = glm::vec4{1.0f,1.0f,1.0f,1.0f});
	// void drawLine(const glm::vec3& p0, const glm::vec3& p1, glm::vec4 color = glm::vec4{1.0f,1.0f,1.0f,1.0f});
private:
	friend class DebugDrawPass;

	auto reset() -> void;

public:
	auto drawBox(const owl::vec3f& midPoint, const owl::vec3f& extent, owl::vec3f color,
				 const owl::LinearSpace3f& orientation) -> void override;

private:
	std::vector<DebugDrawVertex> vertices_;
};
