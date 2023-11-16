#define GLFW_INCLUDE_GLEXT
#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <NullRenderer.h>
#include <RendererBase.h>
#include "FastVoxelTraversalRenderer.h"
#include "NanoViewer.h"

#include <imgui.h>

#include <driver_types.h>
#include <owl/owl.h>

#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_win32.h>

#include <cuda_runtime.h>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

enum class RenderMode
{
	mono,
	stereo
};

struct RendererInitializationInfo
{
	// on stereo we expect that those resources are of a array type in native API
	cudaGraphicsResource_t colorRt;
	cudaGraphicsResource_t minMaxRt;
	RenderMode mode{ RenderMode::mono };

	cudaExternalSemaphore_t waitSemaphore;
	cudaExternalSemaphore_t signalSemaphore;

	cudaUUID_t deviceUuid;
};

class Viewer final : public NanoViewer
{
public:
	explicit Viewer(const std::string& title = "Sample Viewer", int initWindowWidth = 1980,
					int initWindowHeight = 1080);

	~Viewer() override;

protected:
	auto render() -> void override;
	auto resize(const owl::vec2i& newSize) -> void override;

	auto cameraChanged() -> void override;

	auto onFrameBegin() -> void override;

	auto gui() -> void override;

private:
	auto selectRenderer(const std::uint32_t index) -> void
	{
		if (selectedRendererIndex_ == index)
		{
			return;
		}
		if (currentRenderer_)
		{
			currentRenderer_->deinitialize();
		}

		assert(index < b3d::renderer::registry.size());
		selectedRendererIndex_ = index;
		currentRenderer_ = b3d::renderer::registry[selectedRendererIndex_].rendererInstance;

		currentRenderer_->initialize();
	}

	std::shared_ptr<b3d::renderer::RendererBase> currentRenderer_{ nullptr };
	std::int32_t selectedRendererIndex_{ -1 };
	std::int32_t newSelectedRendererIndex_{ -1 };
	std::vector<std::string> registeredRendererNames_{};

	struct GraphicsResources
	{
		GLuint colorTexture;
		GLuint minMaxTexture;
	} resources_;

	RendererInitializationInfo rendererInfo_{};
	RenderMode mode_{ RenderMode::mono };

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

auto Viewer::onFrameBegin() -> void
{
	if (newSelectedRendererIndex_ != selectedRendererIndex_)
	{
		selectRenderer(newSelectedRendererIndex_);
	}
}

Viewer::~Viewer()
{
	vulkanContext_.device.destroySemaphore(synchronizationResources_.vkSignalSemaphore);
	vulkanContext_.device.destroySemaphore(synchronizationResources_.vkWaitSemaphore);

	// TODO: Wired error happens here
	vulkanContext_.device.destroy();
}

auto Viewer::gui() -> void
{
	ImGui::ShowDemoWindow();
	currentRenderer_->gui();

	ImGui::Begin("Renderer Selection");

	const auto& preview = registeredRendererNames_[selectedRendererIndex_];

	if (ImGui::BeginCombo("Renderer", preview.c_str()))
	{
		for (auto n = 0; n < registeredRendererNames_.size(); n++)
		{
			const auto isSelected = (selectedRendererIndex_ == n);
			if (ImGui::Selectable(registeredRendererNames_[n].c_str(), isSelected))
			{
				newSelectedRendererIndex_ = n;
			}

			if (isSelected)
			{
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}

	ImGui::End();
}

auto Viewer::resize(const owl::vec2i& newSize) -> void
{
	OWLViewer::resize(newSize);
	cameraChanged();
}

auto Viewer::cameraChanged() -> void
{
}

Viewer::Viewer(const std::string& title, const int initWindowWidth, const int initWindowHeight)
	: NanoViewer(title, initWindowWidth, initWindowHeight)
{
	gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
	gladLoadGL();

	vk::DynamicLoader dl;
	auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");

	VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

	constexpr auto instanceExtensions = std::array{ VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME };

	constexpr auto appInfo = vk::ApplicationInfo{ .pApplicationName = "GL_CUDA_interop",
												  .applicationVersion = 1,
												  .pEngineName = "GL_CUDA_interop",
												  .engineVersion = 1,
												  .apiVersion = VK_VERSION_1_3 };

	{
		// ReSharper disable once CppVariableCanBeMadeConstexpr
		const auto instanceCreateInfo = vk::InstanceCreateInfo{ .pApplicationInfo = &appInfo,
																.enabledExtensionCount = instanceExtensions.size(),
																.ppEnabledExtensionNames = instanceExtensions.data() };
		const auto result = vk::createInstance(instanceCreateInfo);
		assert(result.result == vk::Result::eSuccess);
		vulkanContext_.instance = result.value;
	}

	VULKAN_HPP_DEFAULT_DISPATCHER.init(vulkanContext_.instance);

	{
		const auto result = vulkanContext_.instance.enumeratePhysicalDevices();
		assert(result.result == vk::Result::eSuccess);

		const auto& devices = result.value;

		auto cudaDeviceCount = 0;
		cudaGetDeviceCount(&cudaDeviceCount);
		assert(cudaDeviceCount != 0);
		auto cudaProperties = std::vector<cudaDeviceProp>{};
		cudaProperties.resize(cudaDeviceCount);

		for (auto i = 0; i < cudaDeviceCount; i++)
		{
			cudaGetDeviceProperties(&cudaProperties[i], i);
		}

		auto found = false;
		auto uuid = cudaUUID_t{};
		auto index = 0;
		// search for first matching device with cuda
		for (auto i = 0; i < devices.size(); i++)
		{
			const auto& device = devices[i];
			const auto properties =
				device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();

			const auto& idProperties = properties.get<vk::PhysicalDeviceIDProperties>();

			for (auto j = 0; j < cudaProperties.size(); j++)
			{
				const auto isEqual = std::equal(idProperties.deviceUUID.begin(), idProperties.deviceUUID.end(),
												cudaProperties[j].uuid.bytes);
				if (isEqual)
				{
					found = true;
					index = i;
					uuid = cudaProperties[j].uuid;
					break;
				}
			}

			if (found)
			{
				break;
			}
		}

		vulkanContext_.physicalDevice = devices[index];
		rendererInfo_.deviceUuid = uuid;
	}

	{

		constexpr auto deviceExtensions =
			std::array{ VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME };

		// vulkan device requires at lest one queue
		// ReSharper disable once CppVariableCanBeMadeConstexpr
		const auto priority = 1.0f;
		const auto queueCreateInfo = vk::DeviceQueueCreateInfo{ .queueCount = 1, .pQueuePriorities = &priority };
		const auto deviceCreateInfo = vk::DeviceCreateInfo{ .queueCreateInfoCount = 1,
																.pQueueCreateInfos = &queueCreateInfo,
																.enabledExtensionCount = deviceExtensions.size(),
																.ppEnabledExtensionNames = deviceExtensions.data() };
		const auto result = vulkanContext_.physicalDevice.createDevice(deviceCreateInfo);
		assert(result.result == vk::Result::eSuccess);
		vulkanContext_.device = result.value;
		VULKAN_HPP_DEFAULT_DISPATCHER.init(vulkanContext_.device);
	}

	const auto semaphoreCreateInfo = vk::StructureChain{
		vk::SemaphoreCreateInfo{},
		vk::ExportSemaphoreCreateInfo{ .handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32 }
	};

	{
		const auto result = vulkanContext_.device.createSemaphore(semaphoreCreateInfo.get());
		assert(result.result == vk::Result::eSuccess);
		synchronizationResources_.vkSignalSemaphore = result.value;
	}

	{
		const auto result = vulkanContext_.device.createSemaphore(semaphoreCreateInfo.get());
		assert(result.result == vk::Result::eSuccess);
		synchronizationResources_.vkWaitSemaphore = result.value;
	}

	{
		const auto handleInfo =
			vk::SemaphoreGetWin32HandleInfoKHR{ .semaphore = synchronizationResources_.vkSignalSemaphore,
												.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32 };
		const auto result = vulkanContext_.device.getSemaphoreWin32HandleKHR(handleInfo);
		assert(result.result == vk::Result::eSuccess);
		synchronizationResources_.signalSemaphoreHandle = result.value;
	}

	{
		const auto handleInfo =
			vk::SemaphoreGetWin32HandleInfoKHR{ .semaphore = synchronizationResources_.vkWaitSemaphore,
												.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32 };
		const auto result = vulkanContext_.device.getSemaphoreWin32HandleKHR(handleInfo);
		assert(result.result == vk::Result::eSuccess);
		synchronizationResources_.waitSemaphoreHandle = result.value;
	}

	// TODO: error checks for gl functions
	glGenSemaphoresEXT(1, &synchronizationResources_.glSignalSemaphore);
	glGenSemaphoresEXT(1, &synchronizationResources_.glWaitSemaphore);
	glImportSemaphoreWin32HandleEXT(synchronizationResources_.glSignalSemaphore, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
									synchronizationResources_.signalSemaphoreHandle);
	glImportSemaphoreWin32HandleEXT(synchronizationResources_.glWaitSemaphore, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
									synchronizationResources_.waitSemaphoreHandle);

	auto externalSemaphoreHandleDesc = cudaExternalSemaphoreHandleDesc{};
	externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	externalSemaphoreHandleDesc.flags = 0;
	{
		externalSemaphoreHandleDesc.handle.win32.handle = synchronizationResources_.waitSemaphoreHandle;
		const auto error = cudaImportExternalSemaphore(&rendererInfo_.waitSemaphore, &externalSemaphoreHandleDesc);
		assert(error == cudaError::cudaSuccess);
	}
	{
		externalSemaphoreHandleDesc.handle.win32.handle = synchronizationResources_.signalSemaphoreHandle;
		const auto result = cudaImportExternalSemaphore(&rendererInfo_.signalSemaphore, &externalSemaphoreHandleDesc);
		assert(result == cudaError::cudaSuccess);
	}

	glGenTextures(1, &resources_.colorTexture);
	glBindTexture(GL_TEXTURE_2D_ARRAY, resources_.colorTexture);
	glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, 64, 64, 2);

	glGenTextures(1, &resources_.minMaxTexture);
	glBindTexture(GL_TEXTURE_2D_ARRAY, resources_.minMaxTexture);
	glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RG32F, 64, 64, 2);

	{
		const auto error = cudaGraphicsGLRegisterImage(&rendererInfo_.colorRt, resources_.colorTexture,
													   mode_ == RenderMode::mono ? GL_TEXTURE_2D : GL_TEXTURE_2D_ARRAY,
													   cudaGraphicsRegisterFlagsWriteDiscard);
		assert(error == cudaError::cudaSuccess);
	}
	{
		const auto error = cudaGraphicsGLRegisterImage(&rendererInfo_.minMaxRt, resources_.minMaxTexture,
													   mode_ == RenderMode::mono ? GL_TEXTURE_2D : GL_TEXTURE_2D_ARRAY,
													   cudaGraphicsRegisterFlagsWriteDiscard);
		assert(error == cudaError::cudaSuccess);
	}

	rendererInfo_.mode = mode_;

	// NOTE: rendererInfo will be fed into renderer initialization

	selectRenderer(0);
	newSelectedRendererIndex_ = selectedRendererIndex_;

	for (auto i = 0; i < b3d::renderer::registry.size(); i++)
	{
		registeredRendererNames_.push_back(b3d::renderer::registry[i].name);
	}
}

auto Viewer::render() -> void
{

	const auto view = b3d::renderer::View{ .camera1 = b3d::renderer::Camera{
									 .origin = camera.getFrom(),
									 .at = camera.getAt(),
									 .up = camera.getUp(),
									 .cosFoV = camera.getCosFovy(),
								 } };

	currentRenderer_->render(view);
}

auto main(int argc, char** argv) -> int
{

	b3d::renderer::registerRenderer<b3d::renderer::NullRenderer>("nullRenderer");
	b3d::renderer::registerRenderer<b3d::renderer::FastVoxelTraversalRenderer>("FastVoxelTraversalRenderer");


	std::cout << b3d::renderer::registry.front().name << std::endl;
	using namespace std::string_literals;
	Viewer viewer("Default Viewer"s);

	viewer.enableFlyMode();
	viewer.enableInspectMode(owl::box3f(owl::vec3f(-1.f), owl::vec3f(+1.f)));
	viewer.showAndRunWithGui();
}
