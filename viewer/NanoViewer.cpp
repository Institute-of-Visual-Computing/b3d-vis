#include "glad/glad.h"

#include "NanoViewer.h"

#include "passes/DebugDrawPass.h"
#include "passes/FullscreenTexturePass.h"
#include "passes/InfinitGridPass.h"

#include <Logging.h>

#include <format>
#include <string>

#include "GLUtils.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "owl/helper/cuda.h"


using namespace owl;
using namespace owl::viewer;

namespace
{
	auto reshape(GLFWwindow* window, const int width, const int height) -> void
	{
		auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
		assert(gw);
		gw->resize(vec2i(width, height));
	}

	auto keyboardKey(GLFWwindow* window, const unsigned int key) -> void
	{
		auto& io = ImGui::GetIO();
		if (!io.WantCaptureKeyboard)
		{
			auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
			assert(gw);
			gw->key(key, gw->getMousePos());
		}
	}

	auto keyboardSpecialKey(GLFWwindow* window, const int key, [[maybe_unused]] int scancode, const int action,
							const int mods) -> void
	{
		const auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
		assert(gw);
		if (action == GLFW_PRESS)
		{
			gw->special(key, mods, gw->getMousePos());
		}
	}

	auto mouseMotion(GLFWwindow* window, const double x, const double y) -> void
	{
		const auto& io = ImGui::GetIO();
		if (!io.WantCaptureMouse)
		{
			const auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
			assert(gw);
			gw->mouseMotion(vec2i(static_cast<int>(x), static_cast<int>(y)));
		}
	}

	auto mouseButton(GLFWwindow* window, const int button, const int action, const int mods) -> void
	{
		const auto& io = ImGui::GetIO();
		if (!io.WantCaptureMouse)
		{
			const auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
			assert(gw);
			gw->mouseButton(button, action, mods);
		}
	}

	auto computeViewProjectionMatrixFromCamera(const Camera& camera, const int width, const int height)
	{

		const auto aspect = width / static_cast<float>(height);

		const auto projectionMatrix = glm::perspective(glm::radians(camera.getFovyInDegrees()), aspect, 0.01f, 10000.0f);
		const auto viewMatrix =
			glm::lookAt(glm::vec3{ camera.position.x, camera.position.y, camera.position.z },
						glm::vec3{ camera.getAt().x, camera.getAt().y, camera.getAt().z },
						glm::normalize(glm::vec3{ camera.getUp().x, camera.getUp().y, camera.getUp().z }));
		const auto viewProjection = projectionMatrix * viewMatrix;
		return viewProjection;
	}

	std::vector<ImFont*> defaultFonts;
	std::unordered_map<float, int> scaleToFont{};
	int currentFontIndex{0};

	auto windowContentScaleCallback(GLFWwindow* window, float scaleX, float scaleY)
	{
		currentFontIndex = scaleToFont[scaleX];
		const auto dpiScale = scaleX;// / 96;
		ImGui::GetStyle().ScaleAllSizes(dpiScale);
	}

	auto initializeGui(GLFWwindow* window) -> void
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		auto& io = ImGui::GetIO();
		(void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls

		ImGui::StyleColorsDark();


		constexpr auto baseFontSize = 16.0f;

		ImFontConfig config;

		config.OversampleH = 8;
		config.OversampleV = 8;

		auto monitorCount = 0;
		const auto monitors = glfwGetMonitors(&monitorCount);

		for (auto i = 0; i < monitorCount; i++)
		{
			const auto monitor = monitors[i];
			auto scaleX = 0.0f;
			auto scaleY = 0.0f;
			glfwGetMonitorContentScale(monitor, &scaleX, &scaleY);
			const auto dpiScale = scaleX;// / 96;
			config.SizePixels = dpiScale * baseFontSize;
			auto font = io.Fonts->AddFontFromFileTTF("resources/fonts/Roboto-Medium.ttf", dpiScale * baseFontSize, &config);
			defaultFonts.push_back(font);

			scaleToFont[scaleX] = i;
		}


		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init();
	}

	auto deinitializeGui() -> void
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	std::unique_ptr<FullscreenTexturePass> fsPass;
	std::unique_ptr<InfinitGridPass> igPass;
	std::unique_ptr<DebugDrawPass> ddPass;

	std::unique_ptr<DebugDrawList> ddList;

	struct ViewerSettings
	{
		float lineWidth{ 4.0 };
		std::array<float, 3> gridColor{ 0.95f, 0.9f, 0.92f };
		bool enableDebugDraw{ true };
		bool enableGridFloor{ true };
	};

	ViewerSettings viewerSettings{};

} // namespace

auto NanoViewer::gui() -> void
{
	static auto show_demo_window = true;
	ImGui::ShowDemoWindow(&show_demo_window);

	currentRenderer_->gui();
	static auto showViewerSettings = true;
	ImGui::Begin("Viewer Settings", &showViewerSettings, ImGuiWindowFlags_AlwaysAutoResize);

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

	static auto selectedCameraControlIndex = 0;
	static constexpr auto controls = std::array{ "POI", "Fly" };

	if (ImGui::BeginCombo("Camera Control", controls[selectedCameraControlIndex]))
	{
		for (auto i = 0; i < controls.size(); i++)
		{
			const auto isSelected = i == selectedCameraControlIndex;
			if (ImGui::Selectable(controls[i], isSelected))
			{
				selectedCameraControlIndex = i;
				if (i == 0)
				{
					enableInspectMode();
				}
				if (i == 1)
				{
					enableFlyMode();
				}
			}
			if (isSelected)
			{
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}
	ImGui::Separator();

	ImGui::Checkbox("Enable Grid Floor", &viewerSettings.enableGridFloor);

	if (viewerSettings.enableGridFloor)
	{
		ImGui::SeparatorText("Grid Settings");
		ImGui::ColorEdit3("Color", viewerSettings.gridColor.data());
		ImGui::Separator();
	}

	ImGui::Checkbox("Enable Debug Draw", &viewerSettings.enableDebugDraw);

	if (viewerSettings.enableDebugDraw)
	{
		ImGui::SeparatorText("Debug Draw Settings");
		ImGui::SliderFloat("Line Width", &viewerSettings.lineWidth, 1.0f, 10.0f);
		ImGui::Separator();
	}


	ImGui::End();
}

auto NanoViewer::render() -> void
{
	constexpr auto layout = static_cast<GLuint>(GL_LAYOUT_GENERAL_EXT);

	const auto cam = b3d::renderer::Camera{ .origin = camera.getFrom(),
											.at = camera.getAt(),
											.up = camera.getUp(),
											.cosFoV = camera.getCosFovy(),
											.FoV = glm::radians(camera.fovyInDegrees) };

	const auto view = b3d::renderer::View{
		.cameras = { cam, cam },
		.mode = b3d::renderer::RenderMode::mono,
		.colorRt = { cuDisplayTexture, { static_cast<uint32_t>(fbSize.x), static_cast<uint32_t>(fbSize.y), 1 } },
		.minMaxRt = { cuDisplayTexture, { static_cast<uint32_t>(fbSize.x), static_cast<uint32_t>(fbSize.y), 1 } },
	};

	GL_CALL(glSignalSemaphoreEXT(synchronizationResources_.glSignalSemaphore, 0, nullptr, 0, nullptr, &layout));

	currentRenderer_->render(view);

	// NOTE: this function call return error, when the semaphore wasn't used before (or it could be in the wrong initial
	// state)
	GL_CALL(glWaitSemaphoreEXT(synchronizationResources_.glWaitSemaphore, 0, nullptr, 0, nullptr, nullptr));
}
auto NanoViewer::resize(const owl::vec2i& newSize) -> void
{
	OWLViewer::resize(newSize);
	cameraChanged();
}

auto NanoViewer::cameraChanged() -> void
{
	OWLViewer::cameraChanged();
}

auto NanoViewer::onFrameBegin() -> void
{
	if (newSelectedRendererIndex_ != selectedRendererIndex_)
	{
		selectRenderer(newSelectedRendererIndex_);
	}
}

NanoViewer::NanoViewer(const std::string& title, const int initWindowWidth, const int initWindowHeight,
					   bool enableVsync, const int rendererIndex)
	: owl::viewer::OWLViewer(title, owl::vec2i(initWindowWidth, initWindowHeight), true, enableVsync), resources_{},
	  synchronizationResources_{}
{
	gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
	gladLoadGL();

	static vk::DynamicLoader dl;
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

		debugDrawList_ = std::make_unique<DebugDrawList>();
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
	// TODO: add cuda error checks
	GL_CALL(glGenSemaphoresEXT(1, &synchronizationResources_.glSignalSemaphore));

	GL_CALL(glGenSemaphoresEXT(1, &synchronizationResources_.glWaitSemaphore));
	GL_CALL(glImportSemaphoreWin32HandleEXT(synchronizationResources_.glSignalSemaphore,
											GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
											synchronizationResources_.signalSemaphoreHandle));
	GL_CALL(glImportSemaphoreWin32HandleEXT(synchronizationResources_.glWaitSemaphore, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
											synchronizationResources_.waitSemaphoreHandle));

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

	/*glGenTextures(1, &resources_.colorTexture);
	glBindTexture(GL_TEXTURE_2D_ARRAY, resources_.colorTexture);
	glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, 64, 64, 2);

	glGenTextures(1, &resources_.minMaxTexture);
	glBindTexture(GL_TEXTURE_2D_ARRAY, resources_.minMaxTexture);
	glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RG32F, 64, 64, 2);*/


	/*{
		const auto error = cudaGraphicsGLRegisterImage(&rendererInfo_.colorRt, resources_.colorTexture,
													   mode_ == b3d::renderer::RenderMode::mono ? GL_TEXTURE_2D :
	GL_TEXTURE_2D_ARRAY, cudaGraphicsRegisterFlagsWriteDiscard); assert(error == cudaError::cudaSuccess);
	}
	{
		const auto error = cudaGraphicsGLRegisterImage(&rendererInfo_.minMaxRt, resources_.minMaxTexture,
													   mode_ == b3d::renderer::RenderMode::mono ? GL_TEXTURE_2D :
	GL_TEXTURE_2D_ARRAY, cudaGraphicsRegisterFlagsWriteDiscard); assert(error == cudaError::cudaSuccess);
	}*/

	// rendererInfo_.mode = mode_;

	// NOTE: rendererInfo will be fed into renderer initialization

	selectRenderer(rendererIndex);
	newSelectedRendererIndex_ = selectedRendererIndex_;

	for (auto i = 0; i < b3d::renderer::registry.size(); i++)
	{
		registeredRendererNames_.push_back(b3d::renderer::registry[i].name);
	}
}

auto NanoViewer::showAndRunWithGui() -> void
{
	showAndRunWithGui([]() { return true; });
}

auto NanoViewer::showAndRunWithGui(const std::function<bool()>& keepgoing) -> void
{
	gladLoadGL();

	ddList = std::make_unique<DebugDrawList>();
	fsPass = std::make_unique<FullscreenTexturePass>();
	igPass = std::make_unique<InfinitGridPass>();
	ddPass = std::make_unique<DebugDrawPass>(debugDrawList_.get());

	int width, height;
	glfwGetFramebufferSize(handle, &width, &height);
	resize(vec2i(width, height));

	glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
	glfwSetFramebufferSizeCallback(handle, reshape);
	glfwSetMouseButtonCallback(handle, ::mouseButton);
	glfwSetKeyCallback(handle, keyboardSpecialKey);
	glfwSetCharCallback(handle, keyboardKey);
	glfwSetCursorPosCallback(handle, ::mouseMotion);
	glfwSetWindowContentScaleCallback(handle, windowContentScaleCallback);

	initializeGui(handle);
	glfwMakeContextCurrent(handle);

	while (!glfwWindowShouldClose(handle) && keepgoing())
	{
		onFrameBegin();
		glClear(GL_COLOR_BUFFER_BIT);
		static double lastCameraUpdate = -1.f;
		if (camera.lastModified != lastCameraUpdate)
		{
			cameraChanged();
			lastCameraUpdate = camera.lastModified;
		}

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::PushFont(defaultFonts[currentFontIndex]);
		gui();
		ImGui::PopFont();
		ImGui::EndFrame();

		render();

		fsPass->setViewport(fbSize.x, fbSize.y);
		fsPass->setSourceTexture(fbTexture);
		fsPass->execute();

		const auto viewProjectionMatrix = computeViewProjectionMatrixFromCamera(camera, fbSize.x, fbSize.y);

		if (viewerSettings.enableGridFloor)
		{
			igPass->setViewProjectionMatrix(viewProjectionMatrix);
			igPass->setViewport(fbSize.x, fbSize.y);
			igPass->setGridColor(
				glm::vec3{ viewerSettings.gridColor[0], viewerSettings.gridColor[1], viewerSettings.gridColor[2] });
			igPass->execute();
		}

		if (viewerSettings.enableDebugDraw)
		{
			ddPass->setViewProjectionMatrix(viewProjectionMatrix);
			ddPass->setViewport(fbSize.x, fbSize.y);
			ddPass->setLineWidth(viewerSettings.lineWidth);
			ddPass->execute();
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(handle);
		glfwPollEvents();
	}

	deinitializeGui();
	glfwDestroyWindow(handle);
	glfwTerminate();
}
NanoViewer::~NanoViewer()
{
	vulkanContext_.device.destroySemaphore(synchronizationResources_.vkSignalSemaphore);
	vulkanContext_.device.destroySemaphore(synchronizationResources_.vkWaitSemaphore);
	vulkanContext_.device.destroy();
}
auto NanoViewer::selectRenderer(const std::uint32_t index) -> void
{
	assert(index < b3d::renderer::registry.size());
	if (selectedRendererIndex_ == index)
	{
		return;
	}
	if (currentRenderer_)
	{
		currentRenderer_->deinitialize();
	}


	selectedRendererIndex_ = index;
	currentRenderer_ = b3d::renderer::registry[selectedRendererIndex_].rendererInstance;

	const auto debugInfo = b3d::renderer::DebugInitializationInfo{ debugDrawList_ };

	currentRenderer_->initialize(rendererInfo_, debugInfo);
}
