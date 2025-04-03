#include "DebugDrawPass.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "DebugDrawVertex.h"
#include "GLUtils.h"

DebugDrawPass::DebugDrawPass(DebugDrawList* debugDrawList) : viewProjection_{}, debugDrawList_{ debugDrawList }
{
	using namespace std::string_literals;
	const auto vertexShader = "#version 400\n"
							  "uniform mat4 viewProjection;\n"
							  "layout (location = 0) in vec3 position;\n"
							  "layout (location = 1) in vec2 uv;\n"
							  "layout (location = 2) in vec4 color;\n"
							  "out vec2 frag_uv;\n"
							  "out vec4 frag_color;\n"
							  "void main()\n"
							  "{\n"
							  "    frag_uv = uv;\n"
							  "    frag_color = color;\n"
							  "    gl_Position = viewProjection * vec4(position, 1.0f);\n"
							  "}\n"s;

	const auto fragmentShader =
		"#version 400\n"
		"in vec2 frag_uv;\n"
		"in vec4 frag_color;"
		"uniform float lineWidth;\n"
		"layout (location = 0) out vec4 outColor;\n"
		"void main()\n"
		"{\n"
		"    outColor = vec4(frag_color.xyz, frag_color.w*(1.0f -  1.0f/(lineWidth * fwidth(frag_uv.x))*frag_uv.x));\n"
		"}\n"s;

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

	glGenBuffers(1, &vboHandle_);
}

DebugDrawPass::~DebugDrawPass()
{
	glDeleteProgram(program_);
	glDeleteBuffers(1, &vboHandle_);
}

auto DebugDrawPass::execute() const -> void
{
	if (debugDrawList_->vertices_.empty())
	{
		return;
	}
	GLuint vertexArrayObject;
	glGenVertexArrays(1, &vertexArrayObject);

	const auto lastIsEnabledDepthTest = glIsEnabled(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glViewport(0, 0, width_, height_);
	glUseProgram(program_);
	glUniformMatrix4fv(viewProjectionUniformLocation_, 1, GL_FALSE, glm::value_ptr(viewProjection_));
	glUniform1f(lineWidthUniformLocation_, lineWidth_);
	glEnable(GL_MULTISAMPLE);
	glBindVertexArray(vertexArrayObject);
	GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vboHandle_));
	glEnableVertexAttribArray(positionAttributeLocation_);
	glEnableVertexAttribArray(uvAttributeLocation_);
	glEnableVertexAttribArray(colorAttributeLocation_);
	glVertexAttribPointer(positionAttributeLocation_, 3, GL_FLOAT, GL_FALSE, sizeof(DebugDrawVertex),
						  reinterpret_cast<GLvoid*>(offsetof(DebugDrawVertex, position)));
	glVertexAttribPointer(uvAttributeLocation_, 2, GL_FLOAT, GL_FALSE, sizeof(DebugDrawVertex),
						  reinterpret_cast<GLvoid*>(offsetof(DebugDrawVertex, uv)));
	glVertexAttribPointer(colorAttributeLocation_, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(DebugDrawVertex),
						  reinterpret_cast<GLvoid*>(offsetof(DebugDrawVertex, color)));


	GL_CALL(glBufferData(GL_ARRAY_BUFFER, debugDrawList_->vertices_.size() * sizeof(DebugDrawVertex),
						 debugDrawList_->vertices_.data(), GL_STREAM_DRAW));


	GL_CALL(glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(debugDrawList_->vertices_.size())));
	glUseProgram(0);
	if (lastIsEnabledDepthTest)
	{
		glEnable(GL_DEPTH_TEST);
	}
	// glDisable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);
	glDeleteVertexArrays(1, &vertexArrayObject);
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
