#include "InfinitGridPass.h"
#include "GLUtils.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

InfiniteGridPass::InfiniteGridPass() : viewProjection_{}
{
	const auto vertexShader = std::string{ R"(
#version 400
uniform mat4 viewProjection;
out vec2 uv;
void main()
{
	vec3 vertices[4] = vec3[4](vec3(-200.0f, 0.0f, -200.0f), vec3(200.0f, 0.0f, -200.0f), vec3(200.0f, 0.0f, 200.0f), vec3(-200.0f, 0.0f, 200.0f));
	vec2 vertices_coord[4] = vec2[4](vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 1.0f) );
	gl_Position = viewProjection * vec4(vertices[gl_VertexID], 1.0f);
	uv = vertices_coord[gl_VertexID];
}
)" };

	const auto fragmentShader = std::string{ R"(
#version 400
in vec2 uv;
uniform vec3 gridColor;
layout (location = 0) out vec4 outColor;
const float gridRatio = 32.0f;
//https://www.shadertoy.com/view/mdVfWw
float pristineGrid( in vec2 uv, in vec2 ddx, in vec2 ddy, vec2 lineWidth)
{
	vec2 uvDeriv = vec2(length(vec2(ddx.x, ddy.x)), length(vec2(ddx.y, ddy.y)));
	bvec2 invertLine = bvec2(lineWidth.x > 0.5f, lineWidth.y > 0.5f);
	vec2 targetWidth = vec2( invertLine.x ? 1.0f - lineWidth.x : lineWidth.x, invertLine.y ? 1.0f - lineWidth.y : lineWidth.y);
	vec2 drawWidth = clamp(targetWidth, uvDeriv, vec2(0.5f));
	vec2 lineAA = uvDeriv * 1.5f;
	vec2 gridUV = abs(fract(uv) * 2.0f - 1.0f);
	gridUV.x = invertLine.x ? gridUV.x : 1.0f - gridUV.x;
	gridUV.y = invertLine.y ? gridUV.y : 1.0f - gridUV.y;
	vec2 grid2 = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
	
	grid2 *= clamp(targetWidth / drawWidth, 0.0f, 1.0f);
	grid2 = mix(grid2, targetWidth, clamp(uvDeriv * 2.0f - 1.0f, 0.0f, 1.0f));
	grid2.x = invertLine.x ? 1.0f - grid2.x : grid2.x;
	grid2.y = invertLine.y ? 1.0f - grid2.y : grid2.y;
	return mix(grid2.x, 1.0f, grid2.y);
}

float gridTextureGradBox( in vec2 p, in vec2 ddx, in vec2 ddy )
{
	vec2 w = max(abs(ddx), abs(ddy)) + 0.01f;
	vec2 a = p + 0.5f * w;                        
	vec2 b = p - 0.5f * w;           
	vec2 i = (floor(a) + min(fract(a) * gridRatio, 1.0f) - floor(b) -min(fract(b) * gridRatio, 1.0f))/(gridRatio * w);
    return (1.0f- i.x) * (1.0f - i.y);
}

void main()
{
	vec2 uvw = uv * 4000.0f;
	vec2 ddx_uvw = dFdx(uvw);
	vec2 ddy_uvw = dFdy(uvw);
	float alpha = pristineGrid(uvw, ddx_uvw, ddy_uvw, vec2(1.0f / gridRatio));
	alpha = 1.0f - gridTextureGradBox(uvw, ddx_uvw, ddy_uvw);
	outColor = vec4(gridColor, alpha);
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
	gridColorUniformLocation_ = glGetUniformLocation(program_, "gridColor");
}
InfiniteGridPass::~InfiniteGridPass()
{
	glDeleteProgram(program_);
}
auto InfiniteGridPass::execute() -> void
{
	const auto lastIsEnabledDepthTest = glIsEnabled(GL_DEPTH_TEST);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glViewport(0, 0, width_, height_);
	glUseProgram(program_);
	glUniformMatrix4fv(viewProjectionUniformLocation_, 1, GL_FALSE, glm::value_ptr(viewProjection_));
	glUniform3f(gridColorUniformLocation_, gridColor_.x, gridColor_.y, gridColor_.z);
	glDrawArrays(GL_QUADS, 0, 4);
	glUseProgram(0);
	if (lastIsEnabledDepthTest)
	{
		glEnable(GL_DEPTH_TEST);
	}
}
auto InfiniteGridPass::setGridColor(const glm::vec3& color) -> void
{
	gridColor_ = color;
}
auto InfiniteGridPass::setViewport(const int width, const int height) -> void
{
	width_ = width;
	height_ = height;
}
auto InfiniteGridPass::setViewProjectionMatrix(const glm::mat4& viewProjectionMatrix) -> void
{
	viewProjection_ = viewProjectionMatrix;
}
