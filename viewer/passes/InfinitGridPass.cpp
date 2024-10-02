#include "InfinitGridPass.h"
#include "GLUtils.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

InfinitGridPass::InfinitGridPass() : viewProjection_{}
{
	using namespace std::string_literals;
	const auto vertexShader =
		"#version 400\n"
		"uniform mat4 viewProjection;\n"
		"out vec2 uv;\n"
		"void main()\n"
		"{\n"
		"    vec3 vertices[4]=vec3[4](vec3(-200, 0, -200), vec3(200,0, -200), vec3(200,0,200), vec3(-200,0, 200));\n"
		"    vec2 vertices_coord[4]=vec2[4](vec2(0, 0), vec2(1, 0),  vec2(1,1),vec2(0, 1) );\n"
		"    gl_Position = viewProjection * vec4(vertices[gl_VertexID], 1.0f);\n"
		"    uv = vertices_coord[gl_VertexID];\n"
		"}\n"s;

	const auto fragmentShader =
		"#version 400\n"
		"in vec2 uv;\n"
		"uniform vec3 gridColor;\n"
		"layout (location = 0) out vec4 outColor;\n"
		"const float gridRatio = 32.0f;\n"
		"//https://www.shadertoy.com/view/mdVfWw\n"
		"float pristineGrid( in vec2 uv, in vec2 ddx, in vec2 ddy, vec2 lineWidth)\n"
		"{\n"
		"	vec2 uvDeriv = vec2(length(vec2(ddx.x, ddy.x)), length(vec2(ddx.y, ddy.y)));\n"
		"	bvec2 invertLine = bvec2(lineWidth.x > 0.5, lineWidth.y > 0.5);\n"
		"	vec2 targetWidth = vec2(\n"
		"	invertLine.x ? 1.0 - lineWidth.x : lineWidth.x,\n"
		"	invertLine.y ? 1.0 - lineWidth.y : lineWidth.y\n"
		"	);\n"
		"	vec2 drawWidth = clamp(targetWidth, uvDeriv, vec2(0.5));\n"
		"	vec2 lineAA = uvDeriv * 1.5;\n"
		"	vec2 gridUV = abs(fract(uv) * 2.0 - 1.0);\n"
		"	gridUV.x = invertLine.x ? gridUV.x : 1.0 - gridUV.x;\n"
		"	gridUV.y = invertLine.y ? gridUV.y : 1.0 - gridUV.y;\n"
		"	vec2 grid2 = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);\n"
		"	\n"
		"	grid2 *= clamp(targetWidth / drawWidth, 0.0, 1.0);\n"
		"	grid2 = mix(grid2, targetWidth, clamp(uvDeriv * 2.0 - 1.0, 0.0, 1.0));\n"
		"	grid2.x = invertLine.x ? 1.0 - grid2.x : grid2.x;\n"
		"	grid2.y = invertLine.y ? 1.0 - grid2.y : grid2.y;\n"
		"	return mix(grid2.x, 1.0, grid2.y);\n"
		"	}\n"
		"float gridTextureGradBox( in vec2 p, in vec2 ddx, in vec2 ddy )\n"
		"{\n"
		"	vec2 w = max(abs(ddx), abs(ddy)) + 0.01;\n"
		"   vec2 a = p + 0.5*w;                        \n"
		"   vec2 b = p - 0.5*w;           \n"
		"   vec2 i = (floor(a)+min(fract(a)*gridRatio,1.0)-\n"
		"             floor(b)-min(fract(b)*gridRatio,1.0))/(gridRatio*w);\n"
		"   return (1.0-i.x)*(1.0-i.y);\n"
		"}\n"
		"void main()\n"
		"{\n"
		"	vec2 uvw = uv*4000.0f;\n"
		"	vec2 ddx_uvw = dFdx( uvw );\n" 
        "	vec2 ddy_uvw = dFdy( uvw );\n" 
		"   float alpha = pristineGrid(uvw, ddx_uvw, ddy_uvw, vec2(1.0/gridRatio));\n"
		"   alpha = 1.0f - gridTextureGradBox(uvw, ddx_uvw, ddy_uvw);\n"
		"   outColor = vec4(gridColor, alpha);\n"
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
	gridColorUniformLocation_ = glGetUniformLocation(program_, "gridColor");
}
InfinitGridPass::~InfinitGridPass()
{
	glDeleteProgram(program_);
}
auto InfinitGridPass::execute() const -> void
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
auto InfinitGridPass::setGridColor(const glm::vec3& color) -> void
{
	gridColor_ = color;
}
auto InfinitGridPass::setViewport(const int width, const int height) -> void
{
	width_ = width;
	height_ = height;
}
auto InfinitGridPass::setViewProjectionMatrix(const glm::mat4& viewProjectionMatrix) -> void
{
	viewProjection_ = viewProjectionMatrix;
}
