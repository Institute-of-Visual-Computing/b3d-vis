#pragma once

#include <glm/glm.hpp>
#include "DebugDrawList.h"
#include "Pass.h"
#include "glad/glad.h"

class DebugDrawPass final : public Pass
{
public:
	explicit DebugDrawPass(DebugDrawList* debugDrawList);

	~DebugDrawPass() override;

	auto execute() const -> void override;


	auto setViewport(const int width, const int height) -> void;

	auto setViewProjectionMatrix(const glm::mat4& viewProjectionMatrix) -> void;

	auto setLineWidth(float lineWidth) -> void;

private:
	GLuint program_;
	GLuint viewProjectionUniformLocation_;
	GLuint lineWidthUniformLocation_;
	GLuint positionAttributeLocation_;
	GLuint uvAttributeLocation_;
	GLuint colorAttributeLocation_;
	glm::mat4 viewProjection_;
	GLuint vboHandle_;
	int width_{};
	int height_{};

	DebugDrawList& debugDrawList_;

	float lineWidth_{1.0f};
};
