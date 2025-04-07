#pragma once
#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif
#include "Animation.h"
#include "Camera.h"
#include "CameraControllers.h"
#include "GizmoOperationFlags.h"
#include "framework/DockableWindowViewBase.h"
#include "passes/DebugDrawPass.h"
#include "passes/FullscreenTexturePass.h"
#include "passes/InfinitGridPass.h"


#include <array>
#include <cuda_runtime.h>
#include <memory>


class GizmoHelper;

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

	auto enableFrameGraph(const bool enable) -> void
	{
		viewerSettings_.enableFrameGraph = enable;
	}
	auto setRenderVolume(b3d::renderer::RendererBase* renderer, b3d::renderer::RenderingDataWrapper* renderingData)
		-> void;

	auto setInternalRenderingResolutionScale(const float scale = 1.0f) -> void;
	auto getInternalRenderingResolutionScale() const -> float
	{
		return internalResolutionScale_;
	}

private:
	auto drawGizmos(const CameraMatrices& cameraMatrices, const glm::vec2& position, const glm::vec2& size) const
		-> void;
	auto initializeGraphicsResources() -> void;
	auto deinitializeGraphicsResources() -> void;

	auto initializeViewGraphicsResources() -> void;
	auto deinitializeViewGraphicsResources() -> void;
	auto initializeInternalGraphicsResources() -> void;
	auto deinitializeInternalGraphicsResources() -> void;

	auto renderVolume() -> void;

	auto demoMode(const bool enable) -> void;

	enum class CameraControllerType
	{
		orbit = 0,
		fps
	};

	auto getCameraController(const CameraControllerType& type) -> CameraController*
	{
		switch (type)
		{
		case CameraControllerType::orbit:
			return &orbitController_;
		case CameraControllerType::fps:
			return &fpsController_;
		default:
			return nullptr;
		}
	}

	b3d::renderer::RendererBase* renderer_{};
	b3d::renderer::RenderingDataWrapper* renderingData_{};

	Camera camera_{};
	Camera cameraLastFrame_{};

	FirstPersonCameraController fpsController_{};
	OrbitCameraController orbitController_{};

	CameraControllerType cameraControllerType_{ CameraControllerType::orbit };

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
		bool enableFrameGraph{ false };
	};

	ViewerSettings viewerSettings_{};

	struct GraphicsResources
	{
		GLuint internalVolumeTexture{ 0 };
		cudaGraphicsResource_t cuInternalVolumeTexture{ 0 };
		uint32_t* volumeTexturePointer{ nullptr };
		glm::vec2 internalVolumeTextureSize{ 0 };
	};

	float internalResolutionScale_{ 1.0f };

	GraphicsResources internalGraphicsResources_{};

	struct Framebuffer
	{
		GLuint nativeHandle{ 0 };
		GLuint colorAttachment{};
	};

	struct ViewportGraphicsResources
	{
		Framebuffer viewFbo{};
	};
	ViewportGraphicsResources viewGraphicsResources_{};

	GizmoOperationFlags currentGizmoOperation_{ GizmoOperationFlagBits::none };

	std::unique_ptr<FullscreenTexturePass> fullscreenTexturePass_{};
	std::unique_ptr<InfinitGridPass> infiniteGridPass_{};
	std::unique_ptr<DebugDrawPass> debugDrawPass_{};
};
