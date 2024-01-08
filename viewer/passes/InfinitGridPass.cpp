#include "InfinitGridPass.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../GLUtils.h"

InfinitGridPass::InfinitGridPass() : viewProjection_{}
{
	using namespace std::string_literals;
	const auto vertexShader =
		"#version 400\n"
		"uniform mat4 viewProjection;\n"
		"out vec2 uv;\n"
		"out float t;\n"
		"void main()\n"
		"{\n"
		"    vec3 vertices[4]=vec3[4](vec3(-200, 0, -200), vec3(200,0, -200), vec3(200,0,200), vec3(-200,0, 200));\n"
		"    vec2 vertices_coord[4]=vec2[4](vec2(0, 0), vec2(1, 0),  vec2(1,1),vec2(0, 1) );\n"
		"    gl_Position = viewProjection * vec4(vertices[gl_VertexID], 1.0f);\n"
		"    uv = vertices_coord[gl_VertexID];\n"
		"    t = gl_Position.z;\n"
		"}\n"s;

	const auto fragmentShader =
		"#version 400\n"
		"in vec2 uv;\n"
		"in float t;"
		"layout (location = 0) out vec4 outColor;\n"
		"const float N = 100.0;\n"
		"float gridTextureGradBox( in vec2 p, in vec2 ddx, in vec2 ddy )\n"
		"{\n"
		"	 vec2 w = max(abs(ddx), abs(ddy)) + 0.01;\n"
		"    vec2 a = p + 0.5*w;\n"
		"    vec2 b = p - 0.5*w;\n"
		"    vec2 i = (floor(a)+min(fract(a)*N,1.0)-floor(b)-min(fract(b)*N,1.0))/(N*w);\n"
		"    return (1.0-i.x)*(1.0-i.y);\n"
		"}\n"
		"void main()\n"
		"{\n"
		"    float tt = sqrt(t);//max(1.0f,t);//(t*t)/2.0f;\n"
		"    float a = fract(tt);\n"
		"    float l0  = floor(tt);//pow(2, ceil(log( max(1.0f,floor(tt)))/log(2)));;\n"
		"    vec2 uv_l0 = uv*1000.0f;\n"
		"    vec2 uv_l1 = uv*pow(0.5, l0);\n"
		"    float alpha_l0 = 1.0f-gridTextureGradBox(uv_l0, dFdx( uv_l0 )/4.0f, dFdy( uv_l0 )/4.0f);\n"
		"    float alpha_l1 = 1.0f-gridTextureGradBox(uv_l1, dFdx( uv_l1 ), dFdy( uv_l1 ));\n"
		"    float alpha = mix(alpha_l1, alpha_l0, a);\n"
		"    alpha = mix( alpha_l0, 0.9f, 1.0f-exp( 0.001*t*t ) );\n"
		"    outColor = vec4(vec3(0.7f,0.80f,0.7f), alpha);\n"
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
	glDrawArrays(GL_QUADS, 0, 4);
	glUseProgram(0);
	if (lastIsEnabledDepthTest)
	{
		glEnable(GL_DEPTH_TEST);
	}
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
