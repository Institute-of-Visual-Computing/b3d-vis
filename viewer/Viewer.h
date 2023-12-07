#pragma once

#define GLFW_INCLUDE_GLEXT

#include <glad/glad.h>

#include <GLFW/glfw3.h>


#include <RendererBase.h>
#include "NanoViewer.h"

#include <imgui.h>

#include <cuda_runtime.h>
#include <driver_types.h>
#include <owl/owl.h>

#include "Logging.h"
#include "Vulkan.h"
#include "owlViewer/cuda_helper.h"


class Viewer final : public NanoViewer
{
public:
	explicit Viewer(const std::string& title = "Sample Viewer", int initWindowWidth = 1980,
					int initWindowHeight = 1080, bool enableVsync = false, const int rendererIndex = 0);

	~Viewer() override;
	
protected:
	auto render() -> void override;
	auto resize(const owl::vec2i& newSize) -> void override;

	auto cameraChanged() -> void override;

	auto onFrameBegin() -> void override;

	auto gui() -> void override;

	
private:
	auto selectRenderer(const std::uint32_t index) -> void;
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

public:
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
