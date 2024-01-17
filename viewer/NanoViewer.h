#pragma once
#include <owlViewer/OWLViewer.h>
#include "DebugDrawList.h"

#include <RendererBase.h>

#include "GizmoHelper.h"
#include "Vulkan.h"

class NanoViewer final : public owl::viewer::OWLViewer
{
public:
	explicit NanoViewer(const std::string& title = "Sample Viewer", const int initWindowWidth = 1980,
						const int initWindowHeight = 1080, bool enableVsync = false, const int rendererIndex = 0);
	auto showAndRunWithGui() -> void;
	auto showAndRunWithGui(const std::function<bool()>& keepgoing) -> void;
	virtual ~NanoViewer();
private:
	auto selectRenderer(const std::uint32_t index) -> void;
	auto gui() -> void;
	auto render() -> void override;
	auto resize(const owl::vec2i& newSize) -> void override;
	auto cameraChanged() -> void override;
	auto onFrameBegin() -> void;

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

	b3d::renderer::RendererInitializationInfo rendererInfo_{};
	b3d::renderer::RenderMode mode_{ b3d::renderer::RenderMode::mono };

	// NOTICE: OpenGL <-> CUDA synchronization:
	// https://github.com/nvpro-samples/gl_cuda_simple_interop/blob/master/README.md
	struct VulkanContext
	{
		vk::Device device;
		vk::PhysicalDevice physicalDevice;
		vk::Instance instance;
	} vulkanContext_{};
	struct SynchronizationResources
	{
		vk::Semaphore vkWaitSemaphore;
		vk::Semaphore vkSignalSemaphore;
		GLuint glWaitSemaphore;
		GLuint glSignalSemaphore;
		HANDLE waitSemaphoreHandle;
		HANDLE signalSemaphoreHandle;
	} synchronizationResources_;
};
