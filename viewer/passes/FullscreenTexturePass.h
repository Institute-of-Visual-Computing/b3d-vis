#pragma once

#include "../Pass.h"
#include "glad/glad.h"

class FullscreenTexturePass final : public Pass
	{
	public:
		FullscreenTexturePass();

		~FullscreenTexturePass() override;

		auto execute() const -> void override;

		auto setSourceTexture(const GLuint texture) -> void
		{
			srcTexture_ = texture;
		}

		auto setViewport(const int width, const int height)
		{
			width_ = width;
			height_ = height;
		}

	private:
		GLuint program_;
		GLuint sourceTextureUniformLocation_;
		GLuint srcTexture_;
		int width_{};
		int height_{};
	};
