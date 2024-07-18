#include "NanoViewer.h"

#include "passes/DebugDrawPass.h"
#include "passes/FullscreenTexturePass.h"
#include "passes/InfinitGridPass.h"

#include "InteropUtils.h"

#include <Logging.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <format>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <owl/helper/cuda.h>
#include <print>
#include <stb_image.h>
#include <string>
#include <tracy/Tracy.hpp>

#include <ImGuizmo.h>

#include <IconsFontAwesome6Brands.h>
#include <IconsLucide.h>

#include <boost/process.hpp>

#include "views/ServerConnectSettingsView.h"

#include <string_view>

#include "GizmoOperationFlags.h"

#include "ApplicationContext.h"
#include "Dockspace.h"
#include "views/VolumeView.h"

using namespace owl;


namespace
{
	ApplicationContext applicationContext{};
	std::unique_ptr<Dockspace> dockspace{};
	std::unique_ptr<VolumeView> volumeView{};


	auto currentGizmoOperation = GizmoOperationFlags(GizmoOperationFlagBits::none);
	auto currentGizmoMode(ImGuizmo::LOCAL);

	auto keyboardKey(GLFWwindow* window, const unsigned int key) -> void
	{
		auto& io = ImGui::GetIO();
		if (!io.WantCaptureKeyboard)
		{
		}
	}

	auto keyboardSpecialKey(GLFWwindow* window, const int key, [[maybe_unused]] int scancode, const int action,
							const int mods) -> void
	{

		if (action == GLFW_PRESS)
		{
		}
	}

	auto mouseMotion(GLFWwindow* window, const double x, const double y) -> void
	{
		const auto& io = ImGui::GetIO();
		if (!io.WantCaptureMouse)
		{
		}
	}

	auto mouseButton(GLFWwindow* window, const int button, const int action, const int mods) -> void
	{
		const auto& io = ImGui::GetIO();
		if (!io.WantCaptureMouse)
		{
		}
		else
		{
		}
	}


	[[nodiscard]] auto requestRequiredDpiScales() -> std::vector<float>
	{
		auto requiredDpiScales = std::vector<float>{};
		auto monitorCount = 0;
		const auto monitors = glfwGetMonitors(&monitorCount);
		if (monitorCount == 0)
		{
			throw std::runtime_error{ "No monitor is connected to the system!" };
		}
		requiredDpiScales.reserve(monitorCount);
		for (auto i = 0; i < monitorCount; i++)
		{
			const auto monitor = monitors[i];
			auto scaleX = 0.0f;
			auto scaleY = 0.0f;
			glfwGetMonitorContentScale(monitor, &scaleX, &scaleY);
			requiredDpiScales.push_back(scaleX);
		}
		return requiredDpiScales;
	}


	auto windowContentScaleCallback([[maybe_unused]] GLFWwindow* window, const float scaleX,
									[[maybe_unused]] float scaleY)
	{
		const auto dpiScales = requestRequiredDpiScales();
		applicationContext.getFontCollection().rebuildFont(dpiScales);
		const auto defaultDpiScale = applicationContext.getFontCollection().getDefaultFontDpiScale();
		ImGui::GetStyle().ScaleAllSizes(defaultDpiScale);
	}

	auto onGLFWErrorCallback(int error, const char* description)
	{
		b3d::renderer::log(std::format("Error: {}\n", description), b3d::renderer::LogLevel::error);
	}


	auto initializeGui(GLFWwindow* window) -> void
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		ImGuizmo::AllowAxisFlip(false);
		auto& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls

		ImGui::StyleColorsDark();

		const auto dpiScales = requestRequiredDpiScales();
		applicationContext.getFontCollection().rebuildFont(dpiScales);


		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init();
	}

	auto deinitializeGui() -> void
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	

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
	static auto showDemoWindow = true;
	ImGui::ShowDemoWindow(&showDemoWindow);

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


	ImGui::Separator();

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
		if (ImGui::IsKeyPressed(ImGuiKey_1, false))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::scale);
		}
		if (ImGui::IsKeyPressed(ImGuiKey_2, false))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::translate);
		}
		if (ImGui::IsKeyPressed(ImGuiKey_3, false))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::rotate);
		}
	}

	ImGui::SeparatorText("NVML Settings");


	static auto enablePersistenceMode{ false };
	static auto enabledPersistenceMode{ false };
	static auto showPermissionDeniedMessage{ false };

	uint32_t clock;
	{
		const auto error = nvmlDeviceGetClockInfo(nvmlDevice_, NVML_CLOCK_SM, &clock);
		assert(error == NVML_SUCCESS);
	}

	ImGui::BeginDisabled(!isAdmin_);
	ImGui::Checkbox(std::format("Max GPU SM Clock [current: {} MHz]", clock).c_str(), &enablePersistenceMode);
	ImGui::EndDisabled();
	if (enablePersistenceMode != enabledPersistenceMode)
	{
		if (enablePersistenceMode)
		{
			const auto error =
				nvmlDeviceSetGpuLockedClocks(nvmlDevice_, static_cast<unsigned int>(NVML_CLOCK_LIMIT_ID_TDP),
											 static_cast<unsigned int>(NVML_CLOCK_LIMIT_ID_TDP));

			enabledPersistenceMode = true;

			assert(error == NVML_SUCCESS);
		}

		else
		{
			const auto error = nvmlDeviceResetGpuLockedClocks(nvmlDevice_);

			enabledPersistenceMode = false;
			enablePersistenceMode = false;
			assert(error == NVML_SUCCESS);
		}
	}


	if (!isAdmin_)
	{
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 0.9f, 0.1f, 0.1f, 1.0f });
		ImGui::TextWrapped("This Application should run in admin mode to apply the effect of this option!");
		ImGui::PopStyleColor();
		ImGui::AlignTextToFramePadding();
	}
	// const auto rr = nvmlDeviceSetPersistenceMode(nvmlDevice, enablePersistenceMode?NVML_FEATURE_ENABLED:
	// NVML_FEATURE_DISABLED);


	ImGui::End();
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
	: resources_{}, renderingData_{}, colorMapResources_{}
{

	glfwSetErrorCallback(onGLFWErrorCallback);

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_VISIBLE, true);

	applicationContext.mainWindowHandle_ = glfwCreateWindow(initWindowWidth, initWindowHeight, title.c_str(), NULL, NULL);
	if (!applicationContext.mainWindowHandle_)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetWindowUserPointer(applicationContext.mainWindowHandle_, this);
	glfwMakeContextCurrent(applicationContext.mainWindowHandle_);
	glfwSwapInterval((enableVsync) ? 1 : 0);

	debugDrawList_ = std::make_unique<DebugDrawList>();
	gizmoHelper_ = std::make_unique<GizmoHelper>();


	nvmlInit();


	{
		const auto error =
			nvmlDeviceGetHandleByIndex(renderingData_.data.rendererInitializationInfo.deviceIndex, &nvmlDevice_);
		assert(error == NVML_SUCCESS);
	}

	{
		const auto error = nvmlDeviceResetGpuLockedClocks(nvmlDevice_);
		if (error == NVML_ERROR_NO_PERMISSION)
		{
			isAdmin_ = false;
		}
		if (error == NVML_SUCCESS)
		{
			isAdmin_ = true;
		}
		assert(error == NVML_SUCCESS || error == NVML_ERROR_NO_PERMISSION);
	}


	gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
	gladLoadGL();


	// Create Colormap and load data from default colormap, if present
	{
		GL_CALL(glGenTextures(1, &colorMapResources_.colormapTexture));
		GL_CALL(glBindTexture(GL_TEXTURE_2D, colorMapResources_.colormapTexture));

		// Setup filtering parameters for display
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
						GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

		// Load default colormap
		auto filePath = std::filesystem::path{ "resources/colormaps" };
		if (std::filesystem::exists(filePath / "defaultColorMap.json"))
		{
			colorMapResources_.colorMap = b3d::tools::colormap::load(filePath / "defaultColorMap.json");

			if (std::filesystem::path(colorMapResources_.colorMap.colorMapFilePath).is_relative())
			{
				filePath /= colorMapResources_.colorMap.colorMapFilePath;
			}
			else
			{
				filePath = colorMapResources_.colorMap.colorMapFilePath;
			}
			int x, y, n;

			const auto bla = stbi_info(filePath.string().c_str(), &x, &y, &n);

			auto data = stbi_loadf(filePath.string().c_str(), &x, &y, &n, 0);

			// Load Colormap
			GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, x, y, 0, GL_RGBA, GL_FLOAT, data));

			stbi_image_free(data);

			renderingData_.data.colorMapTexture.extent =
				b3d::renderer::Extent{ static_cast<uint32_t>(x), static_cast<uint32_t>(y), 1 };
			renderingData_.data.colorMapTexture.nativeHandle =
				reinterpret_cast<void*>(colorMapResources_.colormapTexture);
		}
		else
		{
			GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 512, 1, 0, GL_RGBA, GL_FLOAT, nullptr));
			renderingData_.data.colorMapTexture.extent = b3d::renderer::Extent{ 512, 1, 1 };
		}

		std::string colormaptexturename = "ColorMap";
		GL_CALL(glObjectLabel(GL_TEXTURE, colorMapResources_.colormapTexture, colormaptexturename.length() + 1,
							  colormaptexturename.c_str()));

		// TODO: add cuda error checks
		auto rc =
			cudaGraphicsGLRegisterImage(&colorMapResources_.cudaGraphicsResource, colorMapResources_.colormapTexture,
										GL_TEXTURE_2D, cudaGraphicsRegisterFlagsTextureGather);

		renderingData_.data.colorMapTexture.target = colorMapResources_.cudaGraphicsResource;

		renderingData_.data.coloringInfo =
			b3d::renderer::ColoringInfo{ b3d::renderer::ColoringMode::single, vec4f{ 1, 1, 1, 1 },
										 colorMapResources_.colorMap.firstColorMapYTextureCoordinate };

		renderingData_.data.colorMapInfos =
			b3d::renderer::ColorMapInfos{ &colorMapResources_.colorMap.colorMapNames,
										  colorMapResources_.colorMap.colorMapCount,
										  colorMapResources_.colorMap.firstColorMapYTextureCoordinate,
										  colorMapResources_.colorMap.colorMapHeightNormalized };
	}

	// Transfer function
	{
		GL_CALL(glGenTextures(1, &transferFunctionResources_.transferFunctionTexture));
		GL_CALL(glBindTexture(GL_TEXTURE_2D, transferFunctionResources_.transferFunctionTexture));

		// Setup filtering parameters for display
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
						GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same


		std::array<float, 512> initBufferData;

		std::ranges::fill(initBufferData, 1);
		GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 512, 1, 0, GL_RED, GL_FLOAT, initBufferData.data()));

		std::string transferFunctionBufferName = "TransferFunctionTexture";
		GL_CALL(glObjectLabel(GL_TEXTURE, transferFunctionResources_.transferFunctionTexture,
							  transferFunctionBufferName.length() + 1, transferFunctionBufferName.c_str()));

		cudaError rc = cudaGraphicsGLRegisterImage(
			&transferFunctionResources_.cudaGraphicsResource, transferFunctionResources_.transferFunctionTexture,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsTextureGather | cudaGraphicsRegisterFlagsWriteDiscard);

		renderingData_.data.transferFunctionTexture.extent = { 512, 1, 1 };
		renderingData_.data.transferFunctionTexture.target = transferFunctionResources_.cudaGraphicsResource;
		renderingData_.data.transferFunctionTexture.nativeHandle =
			reinterpret_cast<void*>(transferFunctionResources_.transferFunctionTexture);
	}

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
	showAndRunWithGui([&]() { return isRunning_; });
}

auto NanoViewer::draw() -> void
{
	ZoneScoped;

	// TODO: if windows minimized or not visible -> skip rendering
	onFrameBegin();
	glClear(GL_COLOR_BUFFER_BIT);
	static double lastCameraUpdate = -1.f;

	gizmoHelper_->clear();

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGui::PushFont(applicationContext.getFontCollection().getDefaultFont());
	// TODO: Investigate if this combination is alwys intercepted by OS
	if (ImGui::IsKeyDown(ImGuiMod_Alt) and ImGui::IsKeyPressed(ImGuiKey_F4, false))
	{
		isRunning_ = false;
	}

	static auto connectView =
		ServerConnectSettingsView{ applicationContext, "Server Connect", []() { std::println("submit!!!"); } };

	ImGui::BeginMainMenuBar();
	if (ImGui::BeginMenu("Program"))
	{
		if (ImGui::MenuItem(ICON_LC_UNPLUG " Data Service..", nullptr, nullptr))
		{
		}
		if (ImGui::MenuItem("Server Connection...", nullptr, nullptr))
		{
			connectView.open();
			connectView.reset();
		}

		if (ImGui::MenuItem(ICON_LC_LOG_OUT " Quit", "Alt+F4", nullptr))
		{
			isRunning_ = false;
		}

		ImGui::EndMenu();
	}


	if (ImGui::BeginMenu("Tools"))
	{
		if (ImGui::MenuItem(ICON_LC_BAR_CHART_3 " Histogram", nullptr, nullptr))
		{
		}

		if (ImGui::MenuItem("Transfer Function", nullptr, nullptr))
		{
		}

		ImGui::EndMenu();
	}

	if (ImGui::BeginMenu("Help"))
	{
		const auto url = "https://github.com/Institut-of-Visual-Computing/b3d-vis";

		if (ImGui::MenuItem(ICON_FA_GITHUB " Source Code", nullptr, nullptr))
		{
			auto cmd = "";
#ifdef __APPLE__
#ifdef TARGET_OS_MAC
			cmd = "open";
#endif
#elif __linux__
			cmd = "xdg-open";
#elif _WIN32
			cmd = "start";
#else

#endif
			std::system(std::format("{} {}", cmd, url).c_str());
		}
		ImGui::SeparatorText("Develop Tools");
		ImGui::MenuItem(ICON_LC_BUG " Debug Options");
		ImGui::MenuItem(ICON_LC_CIRCLE_GAUGE " Renderer Profiler");

		ImGui::Separator();
		ImGui::MenuItem("About", nullptr, nullptr);
		ImGui::EndMenu();
	}

	ImGui::EndMainMenuBar();

	dockspace->begin();

	volumeView->draw();

	dockspace->end();

	//const ImGuiViewport* viewport = ImGui::GetMainViewport();
	//ImGui::SetNextWindowPos(viewport->WorkPos);
	//ImGui::SetNextWindowSize(viewport->WorkSize);
	//ImGui::Begin("Editor", 0,
	//			 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
	//				 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus);


	//static ImGuiWindowClass windowClass;
	//static ImGuiID dockspaceId = 0;

	//dockspaceId = ImGui::GetID("mainDock");


	//ImGui::DockSpace(dockspaceId);

	//windowClass.ClassId = dockspaceId;
	//windowClass.DockingAllowUnclassed = true;
	//// windowClass.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_AutoHideTabBar;
	////  ImGuiDockNodeFlags_NoWindowMenuButton;


	//ImGui::End();

	/*ImGui::SetNextWindowClass(&windowClass);
	ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);*/

/*
	ImGui::SetNextWindowClass(&windowClass);
	ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);

	gui();
	windowClass.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_AutoHideTabBar | ImGuiDockNodeFlags_NoUndocking;
	ImGui::SetNextWindowClass(&windowClass);
	ImGui::Begin("VolumeViewport", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse);

	auto viewport3dSize = ImGui::GetContentRegionAvail();
	ImVec2 p = ImGui::GetCursorScreenPos();
	ImGui::SetNextItemAllowOverlap();
	ImGui::InvisibleButton("##volumeViewport", viewport3dSize,
						   ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);


	static auto moveCameraFaster = false;
	auto& io = ImGui::GetIO();


	if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
	{
		moveCameraFaster = true;
	}

	if (ImGui::IsKeyReleased(ImGuiKey_LeftShift))
	{
		moveCameraFaster = false;
	}

	const auto fastSpeed = 25.0f;
	const auto cameraMoveVelocity = 0.0f;
	auto cameraMoveAcceleration = glm::vec3{ 0 };
	const auto maxCameraMoveAcceleration = 1.0f;
	static auto AccelerationExpire = 0.0;
	const auto sensitivity = 0.1f;
	if (ImGui::IsKeyDown(ImGuiKey_W))
	{
		cameraMoveAcceleration = camera_.forward_ * camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f);
	}
	if (ImGui::IsKeyDown(ImGuiKey_S))
	{
		cameraMoveAcceleration =
			-camera_.forward_ * camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f);
	}
	if (ImGui::IsKeyDown(ImGuiKey_A))
	{
		cameraMoveAcceleration = -glm::normalize(glm::cross(camera_.forward_, camera_.getUp())) *
			camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f);
	}
	if (ImGui::IsKeyDown(ImGuiKey_D))
	{
		cameraMoveAcceleration = glm::normalize(glm::cross(camera_.forward_, camera_.getUp())) *
			camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f);
	}

	auto delta = io.MouseDelta;
	delta.x *= -1.0;


	if (ImGui::IsItemActive())
	{
		if (!ImGuizmo::IsUsing())
		{
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
			{
				const auto right = glm::normalize(glm::cross(camera_.forward_, camera_.getUp()));
				cameraMoveAcceleration += -glm::normalize(glm::cross(camera_.forward_, right)) *
					camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f) * delta.y;

				cameraMoveAcceleration +=
					right * camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f) * delta.x;
			}
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
			{

				auto mouseScreenLocation = io.MousePos;
				const auto up = camera_.getUp();
				const auto right = glm::normalize(glm::cross(up, camera_.forward_));

				const auto f = glm::normalize(camera_.forward_ + right * delta.y + up * delta.x);

				auto rotationAxis = glm::normalize(glm::cross(f, camera_.forward_));

				if (glm::length(rotationAxis) >= 0.001f)
				{
					const auto rotation =
						glm::rotate(glm::identity<glm::mat4>(),
									glm::radians(glm::length(glm::vec2{ delta.x, delta.y }) * sensitivity), f);
					camera_.forward_ = glm::normalize(glm::vec3(rotation * glm::vec4(camera_.forward_, 0.0f)));
					camera_.right_ = glm::normalize(glm::cross(camera_.forward_, up));
				}
			}
		}
	}

	camera_.position_ += cameraMoveAcceleration * io.DeltaTime;


	ImGui::SetCursorScreenPos(p);
	ImGui::SetNextItemAllowOverlap();
	ImGui::Image((ImTextureID)viewport3dResources_.framebufferTexture, viewport3dSize, { 0.0f, 1.0f }, { 1.0f, 0.0f });

	if (viewerSettings.enableDebugDraw)
	{
		ImGui::SetNextItemAllowOverlap();
		ImGui::SetCursorScreenPos(p);
		const auto cameraMatrices = computeViewProjectionMatrixFromCamera(camera_, viewport3dSize.x, viewport3dSize.y);
		drawGizmos(cameraMatrices, glm::vec2{ p.x, p.y }, glm::vec2{ viewport3dSize.x, viewport3dSize.y });
	}

	const auto showControls = true;
	if (showControls)
	{
		ImGui::SetNextItemAllowOverlap();

		const auto scale = ImGui::GetWindowDpiScale();
		auto buttonPosition = p + ImVec2(scale * 20, scale * 20);
		ImGui::SetCursorScreenPos(buttonPosition);
		const auto buttonPadding = scale * 4.0f;
		const auto buttonSize = scale * 40;

		const auto activeColor = ImGui::GetStyle().Colors[ImGuiCol_ButtonActive];

		const auto prevOperationState = currentGizmoOperation;

		if (prevOperationState.containsBit(GizmoOperationFlagBits::scale))
		{
			ImGui::PushStyleColor(ImGuiCol_Button, activeColor);
		}

		ImGui::PushFont(applicationContext.getFontCollection().getBigIconsFont());
		if (ImGui::Button(ICON_LC_SCALE_3D "##scale_control_handle", ImVec2{ buttonSize, buttonSize }))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::scale);
		}
		ImGui::PopFont();

		if (ImGui::BeginItemTooltip())
		{
			ImGui::Text("Scale Volume");
			ImGui::TextDisabled("Hotkey: 1");
			ImGui::EndTooltip();
		}

		if (prevOperationState.containsBit(GizmoOperationFlagBits::scale))
		{
			ImGui::PopStyleColor();
		}
		ImGui::SetNextItemAllowOverlap();
		buttonPosition += ImVec2(0, buttonPadding + buttonSize);
		ImGui::SetCursorScreenPos(buttonPosition);

		if (prevOperationState.containsBit(GizmoOperationFlagBits::translate))
		{
			ImGui::PushStyleColor(ImGuiCol_Button, activeColor);
		}
		ImGui::PushFont(applicationContext.getFontCollection().getBigIconsFont());
		if (ImGui::Button(ICON_LC_MOVE_3D "##translate_control_handle", ImVec2{ buttonSize, buttonSize }))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::translate);
		}
		ImGui::PopFont();
		if (ImGui::BeginItemTooltip())
		{
			ImGui::Text("Translate Volume");
			ImGui::TextDisabled("Hotkey: 2");
			ImGui::EndTooltip();
		}

		if (prevOperationState.containsBit(GizmoOperationFlagBits::translate))
		{
			ImGui::PopStyleColor();
		}
		ImGui::SetNextItemAllowOverlap();
		buttonPosition += ImVec2(0, buttonPadding + buttonSize);
		ImGui::SetCursorScreenPos(buttonPosition);

		if (prevOperationState.containsBit(GizmoOperationFlagBits::rotate))
		{
			ImGui::PushStyleColor(ImGuiCol_Button, activeColor);
		}
		ImGui::PushFont(applicationContext.getFontCollection().getBigIconsFont());
		if (ImGui::Button(ICON_LC_ROTATE_3D "##rotate_control_handle", ImVec2{ buttonSize, buttonSize }))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::rotate);
		}
		ImGui::PopFont();
		if (ImGui::BeginItemTooltip())
		{
			ImGui::Text("Rotate Volume");
			ImGui::TextDisabled("Hotkey: 3");
			ImGui::EndTooltip();
		}

		if (prevOperationState.containsBit(GizmoOperationFlagBits::rotate))
		{
			ImGui::PopStyleColor();
		}
	}

	ImGui::End();


	connectView.draw();

	*/
	ImGui::PopFont();
	ImGui::EndFrame();

	ImGui::Render();

	volumeView->renderVolume(currentRenderer_.get(), &renderingData_);
		

	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		GLFWwindow* backup_current_context = glfwGetCurrentContext();
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
		glfwMakeContextCurrent(backup_current_context);
	}

	glfwSwapBuffers(applicationContext.mainWindowHandle_);
	glfwPollEvents();
	FrameMark;
}

auto NanoViewer::showAndRunWithGui(const std::function<bool()>& keepgoing) -> void
{
	gladLoadGL();

	

	int width, height;
	glfwGetFramebufferSize(applicationContext.mainWindowHandle_, &width, &height);
	

	glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
	glfwSetFramebufferSizeCallback(applicationContext.mainWindowHandle_,
								   [](GLFWwindow* window, int width, int height)
								   {
									   auto* viewer = static_cast<NanoViewer*>(glfwGetWindowUserPointer(window));
								
									   viewer->draw();
								   });
	glfwSetMouseButtonCallback(applicationContext.mainWindowHandle_, ::mouseButton);
	glfwSetKeyCallback(applicationContext.mainWindowHandle_, keyboardSpecialKey);
	glfwSetCharCallback(applicationContext.mainWindowHandle_, keyboardKey);
	glfwSetCursorPosCallback(applicationContext.mainWindowHandle_, ::mouseMotion);
	glfwSetWindowContentScaleCallback(applicationContext.mainWindowHandle_, windowContentScaleCallback);

	initializeGui(applicationContext.mainWindowHandle_);


	dockspace = std::make_unique<Dockspace>();
	volumeView = std::make_unique<VolumeView>(applicationContext, dockspace.get());

	glfwMakeContextCurrent(applicationContext.mainWindowHandle_);

	while (!glfwWindowShouldClose(applicationContext.mainWindowHandle_) && keepgoing())
	{
		{
			draw();
		}
	}

	deinitializeGui();
	currentRenderer_->deinitialize();
	glfwDestroyWindow(applicationContext.mainWindowHandle_);
	glfwTerminate();
}
NanoViewer::~NanoViewer()
{
	cudaGraphicsUnregisterResource(transferFunctionResources_.cudaGraphicsResource);
	cudaGraphicsUnregisterResource(colorMapResources_.cudaGraphicsResource);

	if (isAdmin_)
	{
		const auto error = nvmlDeviceResetGpuLockedClocks(nvmlDevice_);
		assert(error == NVML_SUCCESS);
	}
	nvmlShutdown();
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

	const auto debugInfo = b3d::renderer::DebugInitializationInfo{ debugDrawList_, gizmoHelper_ };

	currentRenderer_->initialize(&renderingData_.buffer, debugInfo);
}
