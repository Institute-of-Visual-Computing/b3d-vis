#pragma once
#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif
#include "Camera.h"

#include "Animation.h"
#include "GizmoOperationFlags.h"
#include "framework/DockableWindowViewBase.h"
#include "passes/InfinitGridPass.h"


#include <array>
#include <cuda_runtime.h>
#include <memory>


class GizmoHelper;

class FullscreenTexturePass;
class InfinitGridPass;
class DebugDrawPass;

class DebugDrawList;

namespace b3d::renderer
{
	class RendererBase;
	class RenderingDataWrapper;
} // namespace b3d::renderer

class VolumeView final : public DockableWindowViewBase
{
public:
	VolumeView(ApplicationContext& appContext, Dockspace* dockspace);
	~VolumeView() override;

	auto onDraw() -> void override;
	auto onResize() -> void override;

	struct CameraMatrices
	{
		glm::mat4 view;
		glm::mat4 projection;
		glm::mat4 viewProjection;
	};


	auto setRenderVolume(b3d::renderer::RendererBase* renderer, b3d::renderer::RenderingDataWrapper* renderingData)
		-> void;

private:
	auto drawGizmos(const CameraMatrices& cameraMatrices, const glm::vec2& position, const glm::vec2& size) const
		-> void;
	auto initializeGraphicsResources() -> void;
	auto deinitializeGraphicsResources() -> void;

	auto renderVolume() -> void;

	auto demoMode(const bool enable) -> void;

	b3d::renderer::RendererBase* renderer_{};
	b3d::renderer::RenderingDataWrapper* renderingData_{};

	Camera camera_{};
	Camera cameraLastFrame_{};

	animation::PropertyAnimator animator_;
	bool demoModeEnabled_{ false };

	struct CameraFlyAroundAnimationSetting
	{
		glm::vec3 origin{};
		glm::vec3 target{ 0.0f, 0.0f, 0.0f };
		float height{ 1.0f };
		float radius{ 1.0f };
		float stiffness{ 12.0f };
		float dumping{ 1.0f };
	};

	CameraFlyAroundAnimationSetting flyAnimationSettings_ = {};

	struct ViewerSettings
	{
		float lineWidth{ 4.0 };
		std::array<float, 3> gridColor{ 0.95f, 0.9f, 0.92f };
		bool enableDebugDraw{ true };
		bool enableGridFloor{ true };
		bool enableControlToolBar{ true };
	};

	ViewerSettings viewerSettings_{};

	struct GraphicsResources
	{
		GLuint framebuffer;
		GLuint framebufferTexture{ 0 };


		cudaGraphicsResource_t cuFramebufferTexture{ 0 };
		uint32_t* framebufferPointer{ nullptr };
		glm::vec2 framebufferSize{ 0 };
	};


	GraphicsResources graphicsResources_{};

	GizmoOperationFlags currentGizmoOperation_{ GizmoOperationFlagBits::none };

	std::unique_ptr<FullscreenTexturePass> fullscreenTexturePass_;
	std::unique_ptr<InfinitGridPass> infinitGridPass_{};
	std::unique_ptr<DebugDrawPass> debugDrawPass_{};
};
