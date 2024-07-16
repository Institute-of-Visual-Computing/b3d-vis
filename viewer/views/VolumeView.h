#pragma once
#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif
#include "imgui.h"
#include "../Camera.h"
#include "../DockableWindowViewBase.h"
#include "../GLUtils.h"
#include "../GizmoOperationFlags.h"

#include <cuda_runtime.h>
#include <memory>
#include <array>

class GizmoHelper;

class VolumeView final : public DockableWindowViewBase
{
public:
	VolumeView(Dockspace* dockspace);

	auto onDraw() -> void override;

	struct CameraMatrices
	{
		glm::mat4 view;
		glm::mat4 projection;
		glm::mat4 viewProjection;
	};

private:

	auto drawGizmos(const CameraMatrices& cameraMatrices, const glm::vec2& position, const glm::vec2& size) -> void;

	Camera camera_{};

	struct ViewerSettings
	{
		float lineWidth{ 4.0 };
		std::array<float, 3> gridColor{ 0.95f, 0.9f, 0.92f };
		bool enableDebugDraw{ true };
		bool enableGridFloor{ true };
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

	GizmoOperationFlags currentGizmoOperation_{ (int)GizmoOperationFlagBits::none };
	std::shared_ptr<GizmoHelper> gizmoHelper_{};
};
