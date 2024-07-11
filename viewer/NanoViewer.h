#pragma once
#include "GLUtils.h"

#include <GLFW/glfw3.h>
#include <RendererBase.h>
#include <nvml.h>

#include "Camera.h"
#include "ColorMap.h"
#include "DebugDrawList.h"
#include "GizmoHelper.h"

class NanoViewer final
{
public:
	explicit NanoViewer(const std::string& title = "Sample Viewer", const int initWindowWidth = 1980,
						const int initWindowHeight = 1080, bool enableVsync = false, const int rendererIndex = 0);
	auto showAndRunWithGui() -> void;
	auto showAndRunWithGui(const std::function<bool()>& keepgoing) -> void;
	[[nodiscard]] auto getCamera() -> ::Camera&
	{
		return camera_;
	}
	virtual ~NanoViewer();

private:
	auto selectRenderer(const uint32_t index) -> void;
	auto gui() -> void;
	auto render() -> void;
	auto draw() -> void;
	auto resize(const int width, const int height) -> void;
	auto onFrameBegin() -> void;


	struct CameraMatrices
	{
		glm::mat4 view;
		glm::mat4 projection;
		glm::mat4 viewProjection;
	};

	auto drawGizmos(const CameraMatrices& cameraMatrices, const glm::vec2& position, const glm::vec2& size) -> void;
	static auto computeViewProjectionMatrixFromCamera(const ::Camera& camera, const int width, const int height)
		-> CameraMatrices;


	std::shared_ptr<DebugDrawList> debugDrawList_{};
	std::shared_ptr<GizmoHelper> gizmoHelper_{};

	std::shared_ptr<b3d::renderer::RendererBase> currentRenderer_{ nullptr };
	std::int32_t selectedRendererIndex_{ -1 };
	std::int32_t newSelectedRendererIndex_{ -1 };
	std::vector<std::string> registeredRendererNames_{};

	struct GraphicsResources
	{
		GLuint colorTexture;
		GLuint minMaxTexture;

	} resources_;

	struct ColorMapResources
	{
		b3d::tools::colormap::ColorMap colorMap;
		GLuint colormapTexture;
		cudaGraphicsResource_t cudaGraphicsResource;
	} colorMapResources_;

	struct TransferFunctionResources
	{
		GLuint transferFunctionTexture;
		cudaGraphicsResource_t cudaGraphicsResource;
	} transferFunctionResources_;


	b3d::renderer::RenderingDataWrapper renderingData_;

	b3d::renderer::RenderMode mode_{ b3d::renderer::RenderMode::mono };

	nvmlDevice_t nvmlDevice_{};
	bool isAdmin_{ false };

	struct Viewport3dResources
	{
		GLuint framebuffer;


		GLuint framebufferTexture{ 0 };


		cudaGraphicsResource_t cuFramebufferTexture{ 0 };
		uint32_t* framebufferPointer{ nullptr };
		glm::vec2 framebufferSize{ 0 };
	};

	Viewport3dResources viewport3dResources_;

	auto initializeViewport3dResources(const int width, const int height) -> void;
	auto updateViewport3dResources() -> void;
	auto renderViewport3d(const int width, const int height) -> void;
	auto cleanupViewport3DResources() -> void;


	GLFWwindow* handle_{ nullptr };

	::Camera camera_{};

	bool isRunning_{true};
};
