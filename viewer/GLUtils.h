#pragma once

#include <assert.h>
#include <format>
#include <vector>

#include "Logging.h"
#include "glad/glad.h"

#define GL_CALL(_CALL)                                                                                                 \
	do                                                                                                                 \
	{                                                                                                                  \
		_CALL;                                                                                                         \
		GLenum gl_err = glGetError();                                                                                  \
		if (gl_err != 0)                                                                                               \
			b3d::renderer::log(std::format("[NanoRenderer] GL error 0x{} returned from '{}'.\n", gl_err, #_CALL).c_str());            \
	}                                                                                                                  \
	while (0)


inline auto shaderCompilationCheck(GLuint handle) -> void
{
	auto status = GLint{ 0 };
	auto logLength = GLint{ 0 };
	glGetShaderiv(handle, GL_COMPILE_STATUS, &status);
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLength);

	if (static_cast<GLboolean>(status) == GL_FALSE)
	{
		b3d::renderer::log("[NanoRenderer] ERROR: failed to compile!");
	}
	if (logLength > 1)
	{
		std::vector<char> buffer;
		buffer.resize((int)(logLength + 1));
		glGetShaderInfoLog(handle, logLength, nullptr, (GLchar*)buffer.data());

		b3d::renderer::log(std::format("[NanoRenderer] {}", buffer.data()).c_str());
	}
	assert(static_cast<GLboolean>(status) != GL_FALSE);
}
