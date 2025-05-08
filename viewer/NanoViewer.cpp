#include "NanoViewer.h"

#include "passes/DebugDrawPass.h"

#include "InteropUtils.h"

#include <GLFW/glfw3.h>


#include <Logging.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <format>
#include <print>
#include <string>


#include <stb_image.h>
#include <tracy/Tracy.hpp>

#include <ImGuizmo.h>

#include <IconsFontAwesome6Brands.h>
#include <IconsLucide.h>

#include <boost/process.hpp>

#include "GizmoOperationFlags.h"

#include <imspinner.h>

#include "features/serverConnect/ServerConnectSettingsView.h"
#include "framework/ModalViewBase.h"

#pragma warning(push, 0)
#include <ImGuiFileDialog.h>
#pragma warning(pop)

#include "Color.h"
#include "Mathematics.h"

#ifdef WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dwmapi.h>
#endif

using namespace owl;

namespace
{
	constexpr std::array colors = { legit::Colors::turqoise,  legit::Colors::greenSea,	  legit::Colors::emerald,
									legit::Colors::nephritis, legit::Colors::peterRiver,  legit::Colors::belizeHole,
									legit::Colors::amethyst,  legit::Colors::wisteria,	  legit::Colors::sunFlower,
									legit::Colors::orange,	  legit::Colors::carrot,	  legit::Colors::pumpkin,
									legit::Colors::alizarin,  legit::Colors::pomegranate, legit::Colors::clouds,
									legit::Colors::silver };


	auto setupGuiStyle() -> void
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
		ImGui::GetStyle().WindowMenuButtonPosition = ImGuiDir_None;

#ifdef WIN32
		winrt::Windows::UI::ViewManagement::UISettings const ui_settings{};
		const auto accentColor =
			Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::Accent) };
		const auto accentColorLight1 =
			Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentLight1) };
		const auto accentColorLight2 =
			Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentLight2) };
		const auto accentColorLight3 =
			Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentLight3) };
		const auto accentColorDark1 =
			Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentDark1) };
		const auto accentColorDark2 =
			Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentDark2) };
		const auto accentColorDark3 =
			Color{ ui_settings.GetColorValue(winrt::Windows::UI::ViewManagement::UIColorType::AccentDark3) };
#else
		const auto accentColor = Color{ 0.0332f, 0.19141f, 0.69531f };
		const auto accentColorLight1 = Color{ 0.09082f, 0.33203f, 0.77734f };
		const auto accentColorLight2 = Color{ 0.21582f, 0.48438f, 0.85547f };
		const auto accentColorLight3 = Color{ 0.41797f, 0.6875f, 1.0f };
		const auto accentColorDark1 = Color{ 0.01685f, 0.10205f, 0.34766f };
		const auto accentColorDark2 = Color{ 0.00854f, 0.05469f, 0.18164f };
		const auto accentColorDark3 = Color{ 0.00275f, 0.01941f, 0.05469f };
#endif
		const auto& brush = ApplicationContext::getStyleBrush();
		auto& style = ImGui::GetStyle();

		style.WindowRounding = 8.0f;
		style.FrameRounding = 4.0f;
		style.GrabRounding = 4.0f;
		style.ScrollbarRounding = 4.0f;

		style.WindowBorderSize = 0.0f;
		style.FrameBorderSize = 2.0f;

		style.WindowPadding = Vector2{ 12.0f, 12.0f };
		style.FramePadding = Vector2{ 8.0f, 4.0f };
		style.ItemSpacing = Vector2{ 8.0f, 12.0f };
		style.ItemInnerSpacing = Vector2{ 0.0f, 0.0f };
		style.IndentSpacing = 20.0f;
		style.ScrollbarSize = 12.0f;
		style.GrabMinSize = 20.0f;

		style.TabRounding = 8.0f;
		style.TabBorderSize = 2.0f;
		style.TabBarBorderSize = 0.0f;
		style.TabBarOverlineSize = 2.0f;

		style.PopupRounding = 8.0f;
		style.PopupBorderSize = 2.0f;

		style.DisabledAlpha = 1.0f;

		auto& styleColors = style.Colors;
		styleColors[ImGuiCol_Text] = static_cast<ImVec4>(Color{ 0.95f, 0.95f, 0.95f, 1.00f });
		styleColors[ImGuiCol_ChildBg] = static_cast<ImVec4>(Color{ 0.15f, 0.15f, 0.15f, 0.85f });
		styleColors[ImGuiCol_Border] = static_cast<ImVec4>(Color{ 0.25f, 0.25f, 0.25f, 0.60f });

		styleColors[ImGuiCol_FrameBg] = static_cast<ImVec4>(Color{ 0.22f, 0.22f, 0.22f, 1.00f });
		styleColors[ImGuiCol_FrameBgHovered] = static_cast<ImVec4>(accentColorDark3);
		styleColors[ImGuiCol_FrameBgActive] = static_cast<ImVec4>(Color{ 0.30f, 0.30f, 0.30f, 1.00f });

		styleColors[ImGuiCol_TextDisabled] = static_cast<ImVec4>(brush.textFillColorDisabledBrush);

		styleColors[ImGuiCol_TitleBg] = static_cast<ImVec4>(brush.acrylicBackgroundFillColorBaseBrush);
		styleColors[ImGuiCol_TitleBgActive] = static_cast<ImVec4>(brush.acrylicBackgroundFillColorBaseBrush);
		styleColors[ImGuiCol_TitleBgCollapsed] = static_cast<ImVec4>(brush.acrylicBackgroundFillColorBaseBrush);
		styleColors[ImGuiCol_Tab] = static_cast<ImVec4>(brush.acrylicBackgroundFillColorBaseBrush);
		styleColors[ImGuiCol_TabDimmed] = static_cast<ImVec4>(brush.acrylicBackgroundFillColorBaseBrush);
		styleColors[ImGuiCol_TabHovered] = static_cast<ImVec4>(brush.acrylicBackgroundFillColorDefaultBrush);
		styleColors[ImGuiCol_TabActive] = static_cast<ImVec4>(brush.solidBackgroundFillColorTertiaryBrush);
		styleColors[ImGuiCol_TabDimmedSelected] = static_cast<ImVec4>(brush.solidBackgroundFillColorTertiaryBrush);
		styleColors[ImGuiCol_TabDimmedSelectedOverline] =
			static_cast<ImVec4>(brush.textControlElevationBorderFocusedBrush);
		styleColors[ImGuiCol_TabSelectedOverline] = static_cast<ImVec4>(brush.textControlElevationBorderFocusedBrush);

		styleColors[ImGuiCol_MenuBarBg] = static_cast<ImVec4>(brush.acrylicBackgroundFillColorBaseBrush);
		styleColors[ImGuiCol_WindowBg] = static_cast<ImVec4>(brush.solidBackgroundFillColorTertiaryBrush);
		styleColors[ImGuiCol_PopupBg] = static_cast<ImVec4>(brush.solidBackgroundFillColorTertiaryBrush);
		styleColors[ImGuiCol_Header] = static_cast<ImVec4>(brush.solidBackgroundFillColorTertiaryBrush);
		styleColors[ImGuiCol_HeaderHovered] = static_cast<ImVec4>(brush.controlFillColorSecondaryBrush);

		styleColors[ImGuiCol_ResizeGrip] = static_cast<ImVec4>(accentColorDark3);
		styleColors[ImGuiCol_ResizeGripActive] = static_cast<ImVec4>(accentColor);
		styleColors[ImGuiCol_ResizeGripHovered] = static_cast<ImVec4>(accentColorLight3);

		styleColors[ImGuiCol_SliderGrab] = static_cast<ImVec4>(accentColor);
		styleColors[ImGuiCol_SliderGrabActive] = static_cast<ImVec4>(accentColorLight3);

		styleColors[ImGuiCol_CheckMark] = static_cast<ImVec4>(accentColor);

		styleColors[ImGuiCol_ScrollbarBg] = static_cast<ImVec4>(Color{ 0.0f, 0.0f, 0.0f, 0.0f });
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

	auto windowContentScaleCallback([[maybe_unused]] GLFWwindow* window, [[maybe_unused]] const float scaleX,
									[[maybe_unused]] float scaleY)
	{
		auto& applicationContext = static_cast<NanoViewer*>(glfwGetWindowUserPointer(window))->getApplicationContext();
		const auto dpiScales = requestRequiredDpiScales();
		applicationContext.getFontCollection().rebuildFont(dpiScales);
		const auto defaultDpiScale = applicationContext.getFontCollection().getDefaultFontDpiScale();
		ImGui::GetStyle().ScaleAllSizes(defaultDpiScale);
	}

	auto onGlfwErrorCallback([[maybe_unused]] int error, const char* description)
	{
		b3d::renderer::log(std::format("Error: {}\n", description), b3d::renderer::LogLevel::error);
	}

	auto initializeGui(GLFWwindow* window, ApplicationContext* applicationContext) -> void
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		ImGuizmo::AllowAxisFlip(false);
		auto& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

		ImGui::StyleColorsDark();

		const auto dpiScales = requestRequiredDpiScales();
		applicationContext->getFontCollection().rebuildFont(dpiScales);


		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init();
		setupGuiStyle();
	}

	auto deinitializeGui() -> void
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	void setUpOpenFileDialogStyle()
	{
		ImGuiFileDialog::Instance()->SetFileStyle(IGFD_FileStyleByExtention, ".fits", ImVec4(2.0f, 7.0f, 2.0f, 0.9f),
												  ICON_LC_BOX);
		ImGuiFileDialog::Instance()->SetFileStyle(IGFD_FileStyleByTypeDir, "", ImVec4(1.0f, 1.0f, 1.0f, 0.9f),
												  ICON_LC_FOLDER);
		ImGuiFileDialog::Instance()->SetFlashingAttenuationInSeconds(1.0f);
	}
} // namespace

auto NanoViewer::onFrameBegin() -> void
{
	if (newSelectedRendererIndex_ != selectedRendererIndex_)
	{
		selectRenderer(newSelectedRendererIndex_);
	}
}

NanoViewer::NanoViewer(const std::string& title, const int initWindowWidth, const int initWindowHeight,
					   const bool enableVerticalSync, const int rendererIndex)
{
	glfwSetErrorCallback(onGlfwErrorCallback);

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_VISIBLE, true);

	const auto mainWindowHandle = glfwCreateWindow(initWindowWidth, initWindowHeight, title.c_str(), nullptr, nullptr);
	if (!mainWindowHandle)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

#ifdef WIN32
#ifndef DWMWA_USE_IMMERSIVE_DARK_MODE
#define DWMWA_USE_IMMERSIVE_DARK_MODE 20
#endif
	const auto hwnd = glfwGetWin32Window(mainWindowHandle);
	BOOL value = TRUE;
	DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &value, sizeof(value));
#endif


	glfwSetWindowUserPointer(mainWindowHandle, this);
	glfwMakeContextCurrent(mainWindowHandle);
	glfwSwapInterval((enableVerticalSync) ? 1 : 0);

	debugDrawList_ = std::make_shared<DebugDrawList>();
	gizmoHelper_ = std::make_shared<GizmoHelper>();

	gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
	gladLoadGL();

	applicationContext_ = std::make_unique<ApplicationContext>();
	applicationContext_->mainWindowHandle_ = mainWindowHandle;

	applicationContext_->setStyleBrush(createDarkThemeBrush(AccentColors{}));

	applicationContext_->setExternalDrawLists(debugDrawList_, gizmoHelper_);

	applicationContext_->setExternalDrawLists(debugDrawList_, gizmoHelper_);

	nvmlInit();

	{
		const auto error =
			nvmlDeviceGetHandleByIndex(renderingData_.data.rendererInitializationInfo.deviceIndex, &nvmlDevice_);
		assert(error == NVML_SUCCESS);
		if (error)
		{
			isAdmin_ = false;
		}
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

	selectRenderer(rendererIndex);
	newSelectedRendererIndex_ = selectedRendererIndex_;

	for (auto i = 0; i < b3d::renderer::registry.size(); i++)
	{
		registeredRendererNames_.push_back(b3d::renderer::registry[i].name);
	}

	setUpOpenFileDialogStyle();
}

auto NanoViewer::run() -> void
{
	run([&]() { return isRunning_; });
}

auto NanoViewer::draw() -> void
{
	ZoneScoped;

	onFrameBegin();
	glClear(GL_COLOR_BUFFER_BIT);

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();

	ImGui::NewFrame();

	applicationContext_->serverClient_.updateServerStatusState(ImGui::GetIO().DeltaTime);
	ImGui::PushFont(applicationContext_->getFontCollection().getDefaultFont());

	if (ImGui::IsKeyDown(ImGuiMod_Alt) and ImGui::IsKeyPressed(ImGuiKey_F4, false))
	{
		isRunning_ = false;
	}

	ImGui::BeginMainMenuBar();
	if (ImGui::BeginMenu("Program"))
	{
		if (ImGui::MenuItem(ICON_LC_UNPLUG " Server Connection...", nullptr, nullptr))
		{
			serverConntect_->open();
			serverConntect_->reset();
		}

		ImGui::EndMenu();
	}


	if (ImGui::BeginMenu("Tools"))
	{
		// TODO: feature request for histogram
		/*if (ImGui::MenuItem(ICON_LC_BAR_CHART_3 " Histogram", nullptr, nullptr))
		{
		}*/

		ImGui::EndMenu();
	}
	ImGui::EndMainMenuBar();

	mainMenu_->draw();

	applicationContext_->getMainDockspace()->begin();

	serverConntect_->draw();

	for (const auto component : applicationContext_->updatableComponents_)
	{
		component->update();
	}

	for (const auto component : applicationContext_->rendererExtensions_)
	{
		component->updateRenderingData(renderingData_);
	}
	volumeView_->draw();

	applicationContext_->getMainDockspace()->end();

	if (showImGuiDemo_)
	{
		ImGui::ShowDemoWindow(&showImGuiDemo_);
	}

	if (showDebugOptions_)
	{
		ImGui::Begin("Debug Options", &showDebugOptions_, ImGuiWindowFlags_AlwaysAutoResize);

		auto scale = volumeView_->getInternalRenderingResolutionScale();
		if (ImGui::SliderFloat("Internal Resolution Scale", &scale, 0.1f, 1.0f))
		{
			volumeView_->setInternalRenderingResolutionScale(scale);
		}

		if (ui::Button("Restore Default Layout", Vector2{ ImGui::GetContentRegionAvail().x, 0.0f }))
		{
			applicationContext_->settings_.restoreDefaultLayoutSettings();
		}

		static bool enableGridFloor = true;

		if (ui::ToggleSwitch(enableGridFloor, "Enable Grid Floor", "", ""))
		{
			enableGridFloor = !enableGridFloor;
			volumeView_->enableInfinitGridFloor(enableGridFloor);
		}

		ImGui::End();
	}

	const auto currentFrameTime = 1.0f / ImGui::GetIO().Framerate;
	const auto maxFrameTimeTarget = currentFrameTime > (1.0f / 60.0f) ? 1.0f / 30.0f : 1.0f / 60.0f;

	applicationContext_->profiler_->collectGpuTimers(currentRenderer_->getGpuTimers().getAllCurrent());
	applicationContext_->profiler_->collectGpuTimers(applicationContext_->getGlGpuTimers().getAllCurrent());

	volumeView_->enableFrameGraph(showProfiler_);

	auto& tasks = applicationContext_->profiler_->gpuProfilerTasks();
	applicationContext_->gpuGraph_.maxFrameTime = maxFrameTimeTarget;
	applicationContext_->gpuGraph_.LoadFrameData(tasks.data(), tasks.size());

	static float maxFrameTime = 0.0f;
	maxFrameTime = maxFrameTime * 0.998f + 0.002f * ImGui::GetIO().DeltaTime;

	constexpr auto frameWidth = 3;
	constexpr auto frameSpacing = 1;
	constexpr auto useColoredLegendText = true;
	applicationContext_->gpuGraph_.frameWidth = frameWidth;
	applicationContext_->gpuGraph_.frameSpacing = frameSpacing;
	applicationContext_->gpuGraph_.maxFrameTime = maxFrameTime * 3.0f;
	applicationContext_->gpuGraph_.useColoredLegendText = useColoredLegendText;

	ImGui::PopFont();
	ImGui::EndFrame();

	ImGui::Render();

	gizmoHelper_->clear();

	const auto& record = applicationContext_->getGlGpuTimers().record("ImGui Pass");
	record.start();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	record.stop();

	if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		const auto backupCurrentContext = glfwGetCurrentContext();
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
		glfwMakeContextCurrent(backupCurrentContext);
	}

	glfwSwapBuffers(applicationContext_->mainWindowHandle_);
	glfwPollEvents();
	applicationContext_->getGlGpuTimers().nextFrame();
	applicationContext_->profiler_->nextFrame();

	FrameMark;
}

auto NanoViewer::run(const std::function<bool()>& keepgoing) -> void
{
	gladLoadGL();

	int width, height;
	glfwGetFramebufferSize(applicationContext_->mainWindowHandle_, &width, &height);

	glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
	glfwSetFramebufferSizeCallback(applicationContext_->mainWindowHandle_,
								   [](GLFWwindow* window, int, int)
								   {
									   auto* viewer = static_cast<NanoViewer*>(glfwGetWindowUserPointer(window));

									   viewer->draw();
								   });

	glfwSetWindowContentScaleCallback(applicationContext_->mainWindowHandle_, windowContentScaleCallback);

	initializeGui(applicationContext_->mainWindowHandle_, applicationContext_.get());

	volumeView_ = std::make_unique<VolumeView>(*applicationContext_, applicationContext_->getMainDockspace());
	volumeView_->setRenderVolume(currentRenderer_.get(), &renderingData_);

	transferMapping_ = std::make_unique<TransferMapping>(*applicationContext_);
	// TODO: we need a system for graphics resource initialization/deinitialization
	transferMapping_->initializeResources();
	transferMapping_->updateRenderingData(renderingData_);

	soFiaSearch_ = std::make_unique<SoFiaSearch>(*applicationContext_);
	soFiaSearch_->initializeResources();
	projectExplorer_ = std::make_unique<ProjectExplorer>(*applicationContext_);

	mainMenu_ = std::make_unique<MenuBar>(*applicationContext_);
	serverConntect_ =
		std::make_unique<ServerConnectSettingsView>(*applicationContext_, "Server Connect", [](ModalViewBase*) {});

	applicationContext_->addMenuBarTray(
		[&]()
		{
			auto icon = ICON_LC_SERVER;
			auto signal = ui::SignalState::success;
			if (applicationContext_->serverClient_.getLastServerStatusState().health ==
				b3d::tools::project::ServerHealthState::ok)
			{
				signal = ui::SignalState::success;
			}
			else if (applicationContext_->serverClient_.getLastServerStatusState().health ==
					 b3d::tools::project::ServerHealthState::testing)
			{
				signal = ui::SignalState::caution;
			}
			else
			{
				signal = ui::SignalState::critical;
				icon = ICON_LC_SERVER_OFF;
			}

			if (ui::SignalButton(signal, icon))
			{
				serverConntect_->open();
				serverConntect_->reset();
			}

			enum class RequestStatus
			{
				pending,
				ready
			};

			struct Request
			{
				int progress{};
				std::string label;
				RequestStatus status{ RequestStatus::pending };
			};

			static auto actualRequests = std::vector<Request>{};

			auto hasPendingRequests = applicationContext_->serverClient_.getLastServerStatusState().busyState ==
				b3d::tools::project::ServerBusyState::processing;

			ImGui::SameLine();
			ImGui::SetNextItemAllowOverlap();
			const auto pos = Vector2{ ImGui::GetCursorPos() };
			const auto spinnerRadius = ImGui::GetFontSize() * 0.25f;
			const auto itemWidth = ImGui::GetStyle().FramePadding.x * 2 + spinnerRadius * 4;

			if (ui::Button("##requestQueue", Vector2{ itemWidth, 0.0f }))
			{
			}
			if (ImGui::IsItemHovered())
			{
			}
			if (hasPendingRequests)
			{
				ImGui::SetCursorPos(pos + Vector2{ ImGui::GetStyle().FramePadding } * 2);
				ImSpinner::SpinnerRotateSegments("abs", spinnerRadius, 2.0f);
			}
			else
			{
				const auto offset = (itemWidth - ImGui::CalcTextSize(ICON_LC_CIRCLE_CHECK).x) * 0.5f;
				ImGui::SetCursorPos(pos + Vector2(offset, 0));
				ImGui::Text(ICON_LC_CIRCLE_CHECK);
			}
		});

	applicationContext_->addMenuToggleAction(
		showProfiler_, [](bool) {}, "Help", ICON_LC_CIRCLE_GAUGE " Renderer Profiler", std::nullopt, "Develop Tools");
	applicationContext_->addMenuToggleAction(
		showDebugOptions_, [](bool) {}, "Help", ICON_LC_BUG " Debug Options", std::nullopt, "Develop Tools");
	applicationContext_->addMenuToggleAction(
		showImGuiDemo_, [](bool) {}, "Help", "ImGui Demo", std::nullopt, "Develop Tools");


	applicationContext_->addMenuAction(
		[]()
		{
			const auto url = "https://github.com/Institut-of-Visual-Computing/b3d-vis";
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
		},
		"Help", ICON_FA_GITHUB " Source Code");


	applicationContext_->addMenuToggleAction(showAboutWindow_, [](bool) {}, "Help", "About");
	applicationContext_->addMenuAction([&]() { isRunning_ = false; }, "Program", "Quit", "Alt+F4", std::nullopt, 100);

	glfwMakeContextCurrent(applicationContext_->mainWindowHandle_);

	while (!glfwWindowShouldClose(applicationContext_->mainWindowHandle_) && keepgoing())
	{
		{
			draw();
		}
	}

	deinitializeGui();
	volumeView_.reset(); // TODO: unify and maybe also provide initialize/deinitialize
	currentRenderer_->deinitialize();
	glfwDestroyWindow(applicationContext_->mainWindowHandle_);
	glfwTerminate();
}
NanoViewer::~NanoViewer()
{
	if (isAdmin_)
	{
		[[maybe_unused]] const auto error = nvmlDeviceResetGpuLockedClocks(nvmlDevice_);
		assert(error == NVML_SUCCESS);
	}
	nvmlShutdown();
}

auto NanoViewer::selectRenderer(const std::uint32_t index) -> void
{
	assert(index < b3d::renderer::registry.size());
	if (static_cast<uint32_t>(selectedRendererIndex_) == index)
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
