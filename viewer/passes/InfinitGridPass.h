#pragma once

#include <glm/glm.hpp>
#include "../Pass.h"
#include "glad/glad.h"

class InfinitGridPass final : public Pass
{
public:
	InfinitGridPass();

	~InfinitGridPass() override;

	auto execute() const -> void override;

	auto setGridColor(const glm::vec3& color) -> void;
	auto setViewport(const int width, const int height) -> void;

	auto setViewProjectionMatrix(const glm::mat4& viewProjectionMatrix) -> void;

private:
	GLuint program_;
	GLuint viewProjectionUniformLocation_;
	GLuint gridColorUniformLocation_;
	glm::mat4 viewProjection_;

	glm::vec3 gridColor_{0.7f, 0.6f, 0.6f};
	int width_{};
	int height_{};
};
