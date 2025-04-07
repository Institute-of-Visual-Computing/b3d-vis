#include "FullscreenTexturePass.h"
#include "GLUtils.h"

FullscreenTexturePass::FullscreenTexturePass() : srcTexture_{ 0 }
{
	const auto vertexShader = std::string{ R"(
#version 400
out vec2 textureCoords;
void main()
{
    vec2 vertices[3]=vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(vertices[gl_VertexID], 0, 1);
    textureCoords = 0.5 * gl_Position.xy + vec2(0.5);
}
)" };

	const auto fragmentShader = std::string{ R"(
#version 400
in vec2 textureCoords;
uniform sampler2D srcTexture;
layout (location = 0) out vec4 outColor;
void main()
{
    outColor = texture2D(srcTexture, textureCoords);
}
)" };

	const auto vertexShaderHandle = glCreateShader(GL_VERTEX_SHADER);
	auto length = vertexShader.length();
	const GLchar* v[1] = { vertexShader.c_str() };
	glShaderSource(vertexShaderHandle, 1, v, reinterpret_cast<const GLint*>(&length));
	glCompileShader(vertexShaderHandle);
	shaderCompilationCheck(vertexShaderHandle);

	const auto fragmentShaderHandle = glCreateShader(GL_FRAGMENT_SHADER);
	length = fragmentShader.length();
	const GLchar* f[1] = { fragmentShader.c_str() };
	glShaderSource(fragmentShaderHandle, 1, f, reinterpret_cast<const GLint*>(&length));
	glCompileShader(fragmentShaderHandle);
	shaderCompilationCheck(fragmentShaderHandle);

	program_ = glCreateProgram();
	glAttachShader(program_, vertexShaderHandle);
	glAttachShader(program_, fragmentShaderHandle);
	glLinkProgram(program_);

	glDetachShader(program_, vertexShaderHandle);
	glDetachShader(program_, fragmentShaderHandle);
	glDeleteShader(vertexShaderHandle);
	glDeleteShader(fragmentShaderHandle);


	sourceTextureUniformLocation_ = glGetUniformLocation(program_, "srcTexture");
}
FullscreenTexturePass::~FullscreenTexturePass()
{
	glDeleteProgram(program_);
}
auto FullscreenTexturePass::execute() -> void
{
	const auto lastIsEnabledDepthTest = glIsEnabled(GL_DEPTH_TEST);
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, width_, height_);
	glUseProgram(program_);
	glActiveTexture(GL_TEXTURE0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, srcTexture_);
	glUniform1i(sourceTextureUniformLocation_, 0);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glUseProgram(0);
	if (lastIsEnabledDepthTest)
	{
		glEnable(GL_DEPTH_TEST);
	}
}
