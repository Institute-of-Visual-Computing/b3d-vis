#define GLFW_INCLUDE_GLEXT
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "NanoViewer.h"
#include <NullRenderer.h>
#include <RendererBase.h>
#include "FastVoxelTraversalRenderer.h"

#include <imgui.h>

#include <owl/owl.h>
#include <driver_types.h>

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
	cudaGraphicsResource_t colorRT;
	cudaGraphicsResource_t minMaxRT;
	RenderMode mode{ RenderMode::mono };

	cudaExternalSemaphore_t waitSemaphore;
	cudaExternalSemaphore_t signalSemaphore;

	cudaUUID_t deviceUUID;
};

struct Viewer : public NanoViewer
{
	Viewer(const std::string& title = "Sample Viewer", const int initWindowWidth = 1980,
	       const int initWindowHeight = 1080);

	virtual ~Viewer() override;

	void render() override;
	void resize(const owl::vec2i& newSize) override;

	void cameraChanged() override;

	void onFrameBegin() override;

	void gui() override;

  private:
	void selectRenderer(const std::uint32_t index)
	{
		if(selectedRendererIndex_ == index)
			return;
		if(currentRenderer_)
		{
			currentRenderer_->deinitialize();
		}

		assert(index < b3d::registry.size());
		selectedRendererIndex_ = index;
		currentRenderer_ = b3d::registry[selectedRendererIndex_].rendererInstance;

		currentRenderer_->initialize();
	}

	std::shared_ptr<b3d::RendererBase> currentRenderer_{ nullptr };
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

	//NOTICE: OpenGL <-> CUDA synchronization: https://github.com/nvpro-samples/gl_cuda_simple_interop/blob/master/README.md
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
	} syncronizationResources_;
};

void Viewer::onFrameBegin()
{
	if(newSelectedRendererIndex_ != selectedRendererIndex_)
	{
		selectRenderer(newSelectedRendererIndex_);
	}
}

Viewer::~Viewer()
{
	vulkanContext_.device.destroySemaphore(syncronizationResources_.vkSignalSemaphore);
	vulkanContext_.device.destroySemaphore(syncronizationResources_.vkWaitSemaphore);

	//TODO: Wired error happens here
	vulkanContext_.device.destroy();
}

void Viewer::gui()
{
	ImGui::ShowDemoWindow();
	currentRenderer_->gui();

	ImGui::Begin("Renderer Selection");

	const auto preview = registeredRendererNames_[selectedRendererIndex_];

	if(ImGui::BeginCombo("Renderer", preview.c_str()))
	{
		for(int n = 0; n < registeredRendererNames_.size(); n++)
		{
			const bool isSelected = (selectedRendererIndex_ == n);
			if(ImGui::Selectable(registeredRendererNames_[n].c_str(), isSelected))
				newSelectedRendererIndex_ = n;

			if(isSelected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}

	ImGui::End();
}

void Viewer::resize(const owl::vec2i& newSize)
{
	OWLViewer::resize(newSize);
	cameraChanged();
}

void Viewer::cameraChanged()
{
}

Viewer::Viewer(const std::string& title, const int initWindowWidth, const int initWindowHeight)
    : NanoViewer(title, initWindowWidth, initWindowHeight)
{
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	gladLoadGL();

	vk::DynamicLoader dl;
	auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");

	VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

	const auto instanceExtensions = std::array{ VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME };

	const auto appInfo = vk::ApplicationInfo{ .pApplicationName = "GL_CUDA_interop",
		                                      .applicationVersion = 1,
		                                      .pEngineName = "GL_CUDA_interop",
		                                      .engineVersion = 1,
		                                      .apiVersion = VK_VERSION_1_3 };

	const auto instanceCreateInfo = vk::InstanceCreateInfo{ .pApplicationInfo = &appInfo,
		                                                    .enabledExtensionCount = instanceExtensions.size(),
		                                                    .ppEnabledExtensionNames = instanceExtensions.data() };

	{
		const auto result = vk::createInstance(instanceCreateInfo);
		assert(result.result == vk::Result::eSuccess);
		vulkanContext_.instance = result.value;
	}

	VULKAN_HPP_DEFAULT_DISPATCHER.init(vulkanContext_.instance);

	{
		const auto result = vulkanContext_.instance.enumeratePhysicalDevices();
		assert(result.result == vk::Result::eSuccess);

		const auto devices = result.value;

		auto cudaDeviceCount = 0;
		cudaGetDeviceCount(&cudaDeviceCount);
		assert(cudaDeviceCount != 0);
		auto cudaProperties = std::vector<cudaDeviceProp>{};
		cudaProperties.resize(cudaDeviceCount);

		for(int i = 0; i < cudaDeviceCount; i++)
		{
			cudaGetDeviceProperties(&cudaProperties[i], i);
		}

		auto found = false;
		auto UUID = cudaUUID_t{};
		auto index = 0;
		// search for first matching device with cuda
		for(int i = 0; i < devices.size(); i++)
		{
			const auto& device = devices[i];
			const auto properties =
			    device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();

			const auto idProperties = properties.get<vk::PhysicalDeviceIDProperties>();

			for(int j = 0; j < cudaProperties.size(); j++)
			{
				const auto isEqual = std::equal(idProperties.deviceUUID.begin(), idProperties.deviceUUID.end(),
				                                cudaProperties[j].uuid.bytes);

				if(isEqual)
				{
					found = true;
					index = i;
					UUID = cudaProperties[j].uuid;
					break;
				}
			}

			if(found)
			{
				break;
			}
		}

		vulkanContext_.physicalDevice = devices[index];
		rendererInfo_.deviceUUID = UUID;
		
	}

	{
		const auto deviceExtensions =
		    std::array{ VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME };

		//vulkan device requires at lest one queue
		const auto priority = 1.0f;
		const auto queueCreateInfo = vk::DeviceQueueCreateInfo
		{
			.queueCount = 1,
			.pQueuePriorities = &priority
		};
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
		syncronizationResources_.vkSignalSemaphore = result.value;
	}

	{
		const auto result = vulkanContext_.device.createSemaphore(semaphoreCreateInfo.get());
		assert(result.result == vk::Result::eSuccess);
		syncronizationResources_.vkWaitSemaphore = result.value;
	}

	{
		const auto handleInfo =
		    vk::SemaphoreGetWin32HandleInfoKHR{ .semaphore = syncronizationResources_.vkSignalSemaphore,
			                                    .handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32 };
		const auto result = vulkanContext_.device.getSemaphoreWin32HandleKHR(handleInfo);
		assert(result.result == vk::Result::eSuccess);
		syncronizationResources_.signalSemaphoreHandle = result.value;
	}

	{
		const auto handleInfo =
		    vk::SemaphoreGetWin32HandleInfoKHR{ .semaphore = syncronizationResources_.vkWaitSemaphore,
			                                    .handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32 };
		const auto result = vulkanContext_.device.getSemaphoreWin32HandleKHR(handleInfo);
		assert(result.result == vk::Result::eSuccess);
		syncronizationResources_.waitSemaphoreHandle = result.value;
	}

	//TODO: error checks for gl fucntions
	glGenSemaphoresEXT(1, &syncronizationResources_.glSignalSemaphore);
	glGenSemaphoresEXT(1, &syncronizationResources_.glWaitSemaphore);
	glImportSemaphoreWin32HandleEXT(syncronizationResources_.glSignalSemaphore, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
	                                syncronizationResources_.signalSemaphoreHandle);
	glImportSemaphoreWin32HandleEXT(syncronizationResources_.glWaitSemaphore, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
	                                syncronizationResources_.waitSemaphoreHandle);

	auto externalSemaphoreHandleDesc = cudaExternalSemaphoreHandleDesc{};
	externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	externalSemaphoreHandleDesc.flags = 0;
	{
		externalSemaphoreHandleDesc.handle.win32.handle = syncronizationResources_.waitSemaphoreHandle;
		const auto error = cudaImportExternalSemaphore(&rendererInfo_.waitSemaphore, &externalSemaphoreHandleDesc);
		assert(error == cudaError::cudaSuccess);
	}
	{
		externalSemaphoreHandleDesc.handle.win32.handle = syncronizationResources_.signalSemaphoreHandle;
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
		const auto error = cudaGraphicsGLRegisterImage(&rendererInfo_.colorRT, resources_.colorTexture,
		                                               mode_ == RenderMode::mono ? GL_TEXTURE_2D : GL_TEXTURE_2D_ARRAY,
		                                               cudaGraphicsRegisterFlagsWriteDiscard);
		assert(error == cudaError::cudaSuccess);
	}
	{
		const auto error = cudaGraphicsGLRegisterImage(&rendererInfo_.minMaxRT, resources_.minMaxTexture,
		                                               mode_ == RenderMode::mono ? GL_TEXTURE_2D : GL_TEXTURE_2D_ARRAY,
		                                               cudaGraphicsRegisterFlagsWriteDiscard);
		assert(error == cudaError::cudaSuccess);
	}
	
	rendererInfo_.mode = mode_;

	// NOTE: rendererInfo will be feeded into renderer initialization

	selectRenderer(0);
	newSelectedRendererIndex_ = selectedRendererIndex_;

	for(int i = 0; i < b3d::registry.size(); i++)
	{
		registeredRendererNames_.push_back(b3d::registry[i].name);
	}
}

void Viewer::render()
{

	const auto view = b3d::View{ .camera1 = b3d::Camera{
		                             .origin = camera.getFrom(),
		                             .at = camera.getAt(),
		                             .up = camera.getUp(),
		                             .cosFoV = camera.getCosFovy(),
		                         } };

	currentRenderer_->render(view);
}

int main(int argc, char** argv)
{
	
	b3d::registerRenderer<b3d::NullRenderer>("nullRenderer");
	b3d::registerRenderer<b3d::FastVoxelTraversalRenderer>("FastVoxelTraversalRenderer");


	std::cout << b3d::registry.front().name << std::endl;
	using namespace std::string_literals;
	Viewer viewer("Default Viewer"s);

	viewer.enableFlyMode();
	viewer.enableInspectMode(owl::box3f(owl::vec3f(-1.f), owl::vec3f(+1.f)));
	viewer.showAndRunWithGui();
}
