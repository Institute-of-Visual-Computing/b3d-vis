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


	auto setViewport(const int width, const int height) -> void;

	auto setViewProjectionMatrix(const glm::mat4& viewProjectionMatrix) -> void;

private:
	GLuint program_;
	GLuint viewProjectionUniformLocation_;
	glm::mat4 viewProjection_;
	int width_{};
	int height_{};
};
