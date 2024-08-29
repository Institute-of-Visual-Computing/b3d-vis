#include "NanoViewer.h"

#include "passes/DebugDrawPass.h"

#include "GLUtils.h"
#include "InteropUtils.h"

#include <GLFW/glfw3.h>

#include <Logging.h>

#include <format>
#include <print>
#include <string>

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <owl/helper/cuda.h>

#include <stb_image.h>
#include <tracy/Tracy.hpp>

#include <ImGuizmo.h>

#include <IconsFontAwesome6Brands.h>
#include <IconsLucide.h>

#include <boost/process.hpp>

#include "GizmoOperationFlags.h"

#include "features/projectExplorer/ProjectExplorer.h"
#include "features/sofiaSearch/SoFiaSearch.h"
#include "features/serverConnect/ServerConnectSettingsView.h"
#include "features/transferMapping/TransferMapping.h"
#include "framework/ApplicationContext.h"
#include "framework/MenuBar.h"
#include "views/VolumeView.h"

#include <imspinner.h>

using namespace owl;

namespace
{
	ApplicationContext applicationContext{};
	std::unique_ptr<VolumeView> volumeView{};
	std::unique_ptr<TransferMapping> transferMapping{};
	std::unique_ptr<ProjectExplorer> projectExplorer{};
	std::unique_ptr<SoFiaSearch> soFiaSearch{};
	std::unique_ptr<MenuBar> mainMenu{};
	b3d::renderer::RenderingDataWrapper renderingData{};


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

	auto onGlfwErrorCallback(int error, const char* description)
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
	}

	ImGui::SeparatorText("NVML Settings");


	static auto enablePersistenceMode{ false };
	static auto enabledPersistenceMode{ false };

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
					   const bool enableVsync, const int rendererIndex)
{
	glfwSetErrorCallback(onGlfwErrorCallback);

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_VISIBLE, true);

	applicationContext.mainWindowHandle_ =
		glfwCreateWindow(initWindowWidth, initWindowHeight, title.c_str(), nullptr, nullptr);
	if (!applicationContext.mainWindowHandle_)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetWindowUserPointer(applicationContext.mainWindowHandle_, this);
	glfwMakeContextCurrent(applicationContext.mainWindowHandle_);
	glfwSwapInterval((enableVsync) ? 1 : 0);

	debugDrawList_ = std::make_shared<DebugDrawList>();
	gizmoHelper_ = std::make_shared<GizmoHelper>();

	applicationContext.setExternalDrawLists(debugDrawList_, gizmoHelper_);

	nvmlInit();

	{
		const auto error =
			nvmlDeviceGetHandleByIndex(renderingData.data.rendererInitializationInfo.deviceIndex, &nvmlDevice_);
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


	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();

	ImGui::NewFrame();
	applicationContext.serverClient.updateServerStatusState(ImGui::GetIO().DeltaTime);

	ImGui::PushFont(applicationContext.getFontCollection().getDefaultFont());
	// TODO: Investigate if this combination is always intercepted by OS
	if (ImGui::IsKeyDown(ImGuiMod_Alt) and ImGui::IsKeyPressed(ImGuiKey_F4, false))
	{
		isRunning_ = false;
	}

	static auto connectView = ServerConnectSettingsView{ applicationContext, "Server Connect",
														 [](ModalViewBase*) { std::println("submit!!!"); } };

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

		ImGui::EndMenu();
	}


	if (ImGui::BeginMenu("Tools"))
	{
		if (ImGui::MenuItem(ICON_LC_BAR_CHART_3 " Histogram", nullptr, nullptr))
		{
		}

		ImGui::EndMenu();
	}

	if (ImGui::BeginMenu("Help"))
	{

		if (ImGui::MenuItem(ICON_FA_GITHUB " Source Code", nullptr, nullptr))
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
		}
		ImGui::SeparatorText("Develop Tools");
		ImGui::MenuItem(ICON_LC_BUG " Debug Options");
		ImGui::MenuItem(ICON_LC_CIRCLE_GAUGE " Renderer Profiler");

		ImGui::Separator();
		ImGui::MenuItem("About", nullptr, nullptr);
		ImGui::EndMenu();
	}
	ImGui::EndMainMenuBar();

	mainMenu->draw();

	applicationContext.getMainDockspace()->begin();

	volumeView->draw();
	connectView.draw();

	for (const auto component : applicationContext.updatableComponents_)
	{
		component->update();
	}

	/*for (auto component : applicationContext.drawableComponents_)
	{
		component->draw();
	}*/

	for (const auto component : applicationContext.rendererExtensions_)
	{
		component->updateRenderingData(renderingData);
	}

	// TODO: IT IS DEPRECATED AND IT WILL BE REMOVED!!!
	currentRenderer_->gui();

	applicationContext.getMainDockspace()->end();

	ImGui::ShowDemoWindow();

	ImGui::PopFont();
	ImGui::EndFrame();

	ImGui::Render();

	gizmoHelper_->clear();


	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		const auto backupCurrentContext = glfwGetCurrentContext();
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
		glfwMakeContextCurrent(backupCurrentContext);
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
								   [](GLFWwindow* window, int, int)
								   {
									   auto* viewer = static_cast<NanoViewer*>(glfwGetWindowUserPointer(window));

									   viewer->draw();
								   });

	glfwSetWindowContentScaleCallback(applicationContext.mainWindowHandle_, windowContentScaleCallback);

	initializeGui(applicationContext.mainWindowHandle_);

	volumeView = std::make_unique<VolumeView>(applicationContext, applicationContext.getMainDockspace());
	volumeView->setRenderVolume(currentRenderer_.get(), &renderingData);

	transferMapping = std::make_unique<TransferMapping>(applicationContext);
	// TODO: we need a system for graphics resource initialization/deinitialization
	transferMapping->initializeResources();
	transferMapping->updateRenderingData(renderingData);

	projectExplorer = std::make_unique<ProjectExplorer>(applicationContext);
	soFiaSearch = std::make_unique<SoFiaSearch>(applicationContext);

	mainMenu = std::make_unique<MenuBar>(applicationContext);

	applicationContext.addMenuBarTray(
		[&]()
		{
			auto icon = ICON_LC_SERVER;
			if (applicationContext.serverClient.getLastServerStatusState() == b3d::tools::project::ServerStatusState::ok
				)
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1, 0.5, 0.1, 1.0 });
			}
			else if (applicationContext.serverClient.getLastServerStatusState() ==
					 b3d::tools::project::ServerStatusState::testing)
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 1, 0.65, 0.0, 1.0 });
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.5, 0.1, 0.1, 1.0 });
				icon = ICON_LC_SERVER_OFF;
			}

			if (ImGui::Button(icon))
			{
				// TODO: Open Server Connection Dialog?
			}
			// TODO: Hover to show Status Tooltip.
			
			ImGui::PopStyleColor();

			const auto sampleRequest = std::vector<std::string>{
				"Ready: SoFiA search [10, 30, 50] [20, 40, 100]",
				"Pending: SoFiA search [110, 130, 150] [120, 140, 1100]",
				"Pending: SoFiA search [210, 230, 250] [220, 240, 2100]",
			};

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

			auto hasPendingRequests = false;
			// update fake requests
			for (auto& request : actualRequests)
			{
				request.progress++;
				if (request.progress >= 1000)
				{
					request.status = RequestStatus::ready;
				}
				else
				{
					hasPendingRequests = true;
				}
			}


			ImGui::SameLine();
			ImGui::SetNextItemAllowOverlap();
			const auto pos = ImGui::GetCursorPos();
			const auto spinnerRadius = ImGui::GetFontSize() * 0.25f;
			const auto itemWidth = ImGui::GetStyle().FramePadding.x * 2 + spinnerRadius * 4;
			if (ImGui::Button("##requestQueue", ImVec2(itemWidth, 32)))
			{
				const auto requestIndex = rand() % sampleRequest.size();
				actualRequests.push_back(Request{ .label = sampleRequest[requestIndex] });
			}
			if (ImGui::IsItemHovered())
			{
				if (ImGui::BeginTooltip())
				{
					ImGui::SetNextItemOpen(true);
					if (ImGui::TreeNode("Server Requests"))
					{

						for (const auto& [progress, label, status] : actualRequests)
						{
							ImGui::BulletText(std::format("{}: {}",
														  status == RequestStatus::pending ? "Pending" : "Ready",
														  label)
												  .c_str());
						}
						ImGui::TreePop();
					}
					ImGui::EndTooltip();
				}
			}

			if (hasPendingRequests)
			{
				ImGui::SetCursorPos(pos + ImGui::GetStyle().FramePadding * 2);
				ImSpinner::SpinnerRotateSegments("abs", spinnerRadius, 2.0f);
			}
			else
			{
				const auto offset = (itemWidth - ImGui::CalcTextSize(ICON_LC_CIRCLE_CHECK).x) * 0.5f;
				ImGui::SetCursorPos(pos + ImVec2(offset, 0));
				ImGui::Text(ICON_LC_CIRCLE_CHECK);
			}
		});


	applicationContext.addMenuAction([&]() { isRunning_ = false; }, "Program", "Quit", "Alt+F4", std::nullopt, 100);

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

	currentRenderer_->initialize(&renderingData.buffer, debugInfo);
}
