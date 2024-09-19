#include "NanoViewer.h"

#include "passes/DebugDrawPass.h"

#include "InteropUtils.h"

#include <GLFW/glfw3.h>


#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <Logging.h>

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

using namespace owl;

namespace
{
	constexpr std::array colors = { legit::Colors::turqoise,  legit::Colors::greenSea,	  legit::Colors::emerald,
									legit::Colors::nephritis, legit::Colors::peterRiver,  legit::Colors::belizeHole,
									legit::Colors::amethyst,  legit::Colors::wisteria,	  legit::Colors::sunFlower,
									legit::Colors::orange,	  legit::Colors::carrot,	  legit::Colors::pumpkin,
									legit::Colors::alizarin,  legit::Colors::pomegranate, legit::Colors::clouds,
									legit::Colors::silver };


	


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
		auto& applicationContext  = static_cast<NanoViewer*>(glfwGetWindowUserPointer(window))->getApplicationContext();
		const auto dpiScales = requestRequiredDpiScales();
		applicationContext.getFontCollection().rebuildFont(dpiScales);
		const auto defaultDpiScale = applicationContext.getFontCollection().getDefaultFontDpiScale();
		ImGui::GetStyle().ScaleAllSizes(defaultDpiScale);
	}

	auto onGlfwErrorCallback(int error, const char* description)
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
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls

		ImGui::StyleColorsDark();

		const auto dpiScales = requestRequiredDpiScales();
		applicationContext->getFontCollection().rebuildFont(dpiScales);


		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init();
	}

	auto deinitializeGui() -> void
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}


} // namespace

auto NanoViewer::gui() -> void
{
	/*ImGui::ShowDemoWindow(&showImGuiDemo);*/

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


	ImGui::Checkbox("Enable Grid Floor", &viewerSettings_.enableGridFloor);

	if (viewerSettings_.enableGridFloor)
	{
		ImGui::SeparatorText("Grid Settings");
		ImGui::ColorEdit3("Color", viewerSettings_.gridColor.data());
		ImGui::Separator();
	}

	ImGui::Checkbox("Enable Debug Draw", &viewerSettings_.enableDebugDraw);

	if (viewerSettings_.enableDebugDraw)
	{
		ImGui::SeparatorText("Debug Draw Settings");
		ImGui::SliderFloat("Line Width", &viewerSettings_.lineWidth, 1.0f, 10.0f);
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

	const auto mainWindowHandle = glfwCreateWindow(initWindowWidth, initWindowHeight, title.c_str(), nullptr, nullptr);
	if (!mainWindowHandle)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}


	glfwSetWindowUserPointer(mainWindowHandle, this);
	glfwMakeContextCurrent(mainWindowHandle);
	glfwSwapInterval((enableVsync) ? 1 : 0);


	/*glfwWindowHint(GLFW_DECORATED, false);
	const auto splashScreenWindowHandle = glfwCreateWindow(initWindowWidth, initWindowHeight, title.c_str(), nullptr,
	nullptr); if (!splashScreenWindowHandle)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetWindowUserPointer(splashScreenWindowHandle, this);
	glfwMakeContextCurrent(splashScreenWindowHandle);
	glfwSwapInterval((enableVsync) ? 1 : 0);*/


	debugDrawList_ = std::make_shared<DebugDrawList>();
	gizmoHelper_ = std::make_shared<GizmoHelper>();

	gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
	gladLoadGL();

#if 0
	GLint numExtensions;
	glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);
	std::cout << "- Extensions" << std::endl;
	for (GLint i = 0; i < numExtensions; i++)
	{
		std::cout << glGetStringi(GL_EXTENSIONS, i) << std::endl;
	}
#endif
	applicationContext_ = std::make_unique<ApplicationContext>();
	applicationContext_->mainWindowHandle_ = mainWindowHandle;
	//	applicationContext->splashScreenWindowHandle_ = splashScreenWindowHandle;

	applicationContext_->setExternalDrawLists(debugDrawList_, gizmoHelper_);

	applicationContext_->setExternalDrawLists(debugDrawList_, gizmoHelper_);

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

	applicationContext_->serverClient_.updateServerStatusState(ImGui::GetIO().DeltaTime);
	ImGui::PushFont(applicationContext_->getFontCollection().getDefaultFont());
	// TODO: Investigate if this combination is always intercepted by OS
	if (ImGui::IsKeyDown(ImGuiMod_Alt) and ImGui::IsKeyPressed(ImGuiKey_F4, false))
	{
		isRunning_ = false;
	}

	static auto connectView = ServerConnectSettingsView{ *applicationContext_, "Server Connect",
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
	ImGui::EndMainMenuBar();

	mainMenu_->draw();

	applicationContext_->getMainDockspace()->begin();

	connectView.draw();

	for (const auto component : applicationContext_->updatableComponents_)
	{
		component->update();
	}

	/*for (auto component : applicationContext->drawableComponents_)
	{
		component->draw();
	}*/

	for (const auto component : applicationContext_->rendererExtensions_)
	{
		component->updateRenderingData(renderingData_);
	}
	volumeView_->draw();

	//// TODO: IT IS DEPRECATED AND IT WILL BE REMOVED!!!
	// currentRenderer_->gui();

	applicationContext_->getMainDockspace()->end();

	if (showImGuiDemo_)
	{
		ImGui::ShowDemoWindow(&showImGuiDemo_);
	}

	const auto currentFrameTime = 1.0f / ImGui::GetIO().Framerate;
	const auto maxFrameTimeTarget = currentFrameTime > (1.0f / 60.0f) ? 1.0f / 30.0f : 1.0f / 60.0f;
#if 0

	
	if (showProfiler)
	{
		

		profilersWindow_.gpuGraph().maxFrameTime = maxFrameTimeTarget;

		if (!profilersWindow_.isProfiling())
		{
			auto profilingData = std::vector<legit::ProfilerTask>{};
			double lastEndTime = 0.0f;
			auto colorIndex = 0;
			{
				const auto& gpuTimers = currentRenderer_->getGpuTimers();
				const auto& currentTimings = gpuTimers.getAllCurrent();

				for (const auto& [name, start, stop] : currentTimings)
				{
					auto profilerTask = legit::ProfilerTask{};
					profilerTask.name = name;
					profilerTask.startTime = start / 1000.0f;
					profilerTask.endTime = stop / 1000.0f;
					profilerTask.color = colors[colorIndex % colors.size()];
					profilingData.push_back(profilerTask);
					colorIndex += 2;

					lastEndTime = glm::max(lastEndTime, profilerTask.endTime);
				}
			}

			{
				const auto& gpuTimers = applicationContext->getGlGpuTimers();
				const auto& currentTimings = gpuTimers.getAllCurrent();

				for (const auto& [name, start, stop] : currentTimings)
				{
					auto profilerTask = legit::ProfilerTask{};
					profilerTask.name = name;
					profilerTask.startTime = lastEndTime + start / 1000.0f;
					profilerTask.endTime = lastEndTime + stop / 1000.0f;
					profilerTask.color = colors[colorIndex % colors.size()];
					profilingData.push_back(profilerTask);
					colorIndex += 2;
				}
			}

			profilersWindow_.gpuGraph().LoadFrameData(profilingData.data(), profilingData.size());
		}

		profilersWindow_.render();
	}
#endif
	applicationContext_->profiler_->collectGpuTimers(currentRenderer_->getGpuTimers().getAllCurrent());
	applicationContext_->profiler_->collectGpuTimers(applicationContext_->getGlGpuTimers().getAllCurrent());

	volumeView_->enableFrameGraph(showProfiler_);

	auto& tasks = applicationContext_->profiler_->gpuProfilerTasks();
	applicationContext_->gpuGraph_.maxFrameTime = maxFrameTimeTarget;
	applicationContext_->gpuGraph_.LoadFrameData(tasks.data(), tasks.size());

	static float maxFrameTime = 0;
	maxFrameTime = maxFrameTime * 0.998 + 0.002 * ImGui::GetIO().DeltaTime;

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

auto NanoViewer::showAndRunWithGui(const std::function<bool()>& keepgoing) -> void
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

	applicationContext_->addMenuBarTray(
		[&]()
		{
			auto icon = ICON_LC_SERVER;
			if (applicationContext_->serverClient_.getLastServerStatusState() ==
				b3d::tools::project::ServerStatusState::ok)
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1, 0.5, 0.1, 1.0 });
			}
			else if (applicationContext_->serverClient_.getLastServerStatusState() ==
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
							ImGui::BulletText(
								std::format("{}: {}", status == RequestStatus::pending ? "Pending" : "Ready", label)
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


	applicationContext_->addMenuToggleAction(
		showAboutWindow_, [](bool) {}, "Help", "About");

	applicationContext_->addMenuAction([&]() { isRunning_ = false; }, "Program", "Quit", "Alt+F4", std::nullopt, 100);
	applicationContext_->addMenuAction([&]() { applicationContext_->settings_.restoreDefaultLayoutSettings(); },
									  "Program", "Restore Layout", "", std::nullopt, 10);


	glfwMakeContextCurrent(applicationContext_->mainWindowHandle_);
	// glfwSetWindowSize(applicationContext->mainWindowHandle_, 1000, 600);


	while (!glfwWindowShouldClose(applicationContext_->mainWindowHandle_) && keepgoing())
	{
		{
			draw();
		}
	}

	deinitializeGui();
	currentRenderer_->deinitialize();
	glfwDestroyWindow(applicationContext_->mainWindowHandle_);
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

	currentRenderer_->initialize(&renderingData_.buffer, debugInfo);
}
