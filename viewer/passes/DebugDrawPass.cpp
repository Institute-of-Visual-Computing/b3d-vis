#include "DebugDrawPass.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "DebugDrawVertex.h"
#include "GLUtils.h"

DebugDrawPass::DebugDrawPass(DebugDrawList* debugDrawList) : viewProjection_{}, debugDrawList_{ debugDrawList }
{
	const auto vertexShader = std::string{ R"(
#version 400
uniform mat4 viewProjection;
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec4 color;
out vec2 frag_uv;
out vec4 frag_color;
void main()
{
    frag_uv = uv;
    frag_color = color;
    gl_Position = viewProjection * vec4(position, 1.0f);
}
)" };

	const auto fragmentShader = std::string{ R"(
#version 400
in vec2 frag_uv;
in vec4 frag_color;
uniform float lineWidth;
layout (location = 0) out vec4 outColor;
void main()
{
    outColor = vec4(frag_color.xyz, frag_color.w*(1.0f -  1.0f/(lineWidth * fwidth(frag_uv.x))*frag_uv.x));
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

	viewProjectionUniformLocation_ = glGetUniformLocation(program_, "viewProjection");
	lineWidthUniformLocation_ = glGetUniformLocation(program_, "lineWidth");

	positionAttributeLocation_ = glGetAttribLocation(program_, "position");
	uvAttributeLocation_ = glGetAttribLocation(program_, "uv");
	colorAttributeLocation_ = glGetAttribLocation(program_, "color");

	const auto defaultBufferSize = 16 * 1024;
	vertexBufferSize_ = defaultBufferSize;
	glCreateBuffers(1, &vertexBufferHandle_);
	glNamedBufferStorage(vertexBufferHandle_, defaultBufferSize, nullptr, GL_DYNAMIC_STORAGE_BIT);

	glCreateVertexArrays(1, &vao_);

	glVertexArrayVertexBuffer(vao_, 0, vertexBufferHandle_, 0, sizeof(DebugDrawVertex));

	glEnableVertexArrayAttrib(vao_, positionAttributeLocation_);
	glVertexArrayAttribBinding(vao_, positionAttributeLocation_, 0);
	glVertexArrayAttribFormat(vao_, positionAttributeLocation_, 3, GL_FLOAT, GL_FALSE,
							  offsetof(DebugDrawVertex, position));

	glEnableVertexArrayAttrib(vao_, uvAttributeLocation_);
	glVertexArrayAttribBinding(vao_, uvAttributeLocation_, 0);
	glVertexArrayAttribFormat(vao_, uvAttributeLocation_, 2, GL_FLOAT, GL_FALSE, offsetof(DebugDrawVertex, uv));

	glEnableVertexArrayAttrib(vao_, colorAttributeLocation_);
	glVertexArrayAttribBinding(vao_, colorAttributeLocation_, 0);
	glVertexArrayAttribFormat(vao_, colorAttributeLocation_, 4, GL_UNSIGNED_BYTE, GL_TRUE,
							  offsetof(DebugDrawVertex, color));
}

DebugDrawPass::~DebugDrawPass()
{
	glDeleteVertexArrays(1, &vao_);
	glDeleteProgram(program_);
	glDeleteBuffers(1, &vertexBufferHandle_);
}

auto DebugDrawPass::execute() -> void
{
	if (debugDrawList_->vertices_.empty())
	{
		return;
	}

	const auto vertexDataSize = debugDrawList_->vertices_.size() * sizeof(DebugDrawVertex);
	if (vertexDataSize > vertexBufferSize_)
	{
		vertexBufferSize_ = vertexBufferSize_ * 2;
		glDeleteBuffers(1, &vertexBufferHandle_);
		glCreateBuffers(1, &vertexBufferHandle_);
		glNamedBufferStorage(vertexBufferHandle_, vertexBufferSize_, nullptr, GL_DYNAMIC_STORAGE_BIT);
		glVertexArrayVertexBuffer(vao_, 0, vertexBufferHandle_, 0, sizeof(DebugDrawVertex));
	}
	glNamedBufferSubData(vertexBufferHandle_, 0, vertexDataSize, debugDrawList_->vertices_.data());

	const auto lastIsEnabledDepthTest = glIsEnabled(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glViewport(0, 0, width_, height_);
	glUseProgram(program_);
	glUniformMatrix4fv(viewProjectionUniformLocation_, 1, GL_FALSE, glm::value_ptr(viewProjection_));
	glUniform1f(lineWidthUniformLocation_, lineWidth_);
	glEnable(GL_MULTISAMPLE);
	glBindVertexArray(vao_);
	GL_CALL(glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(debugDrawList_->vertices_.size())));
	glUseProgram(0);
	if (lastIsEnabledDepthTest)
	{
		glEnable(GL_DEPTH_TEST);
	}
	glDisable(GL_MULTISAMPLE);
	debugDrawList_->reset();
}
auto DebugDrawPass::setViewport(const int width, const int height) -> void
{
	width_ = width;
	height_ = height;
}
auto DebugDrawPass::setViewProjectionMatrix(const glm::mat4& viewProjectionMatrix) -> void
{
	viewProjection_ = viewProjectionMatrix;
}
auto DebugDrawPass::setLineWidth(const float lineWidth) -> void
{
	assert(lineWidth > 0.0f);
	lineWidth_ = lineWidth;
}
