#include "NanoViewer.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
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
		if (io.WantCaptureKeyboard)
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
		if (io.WantCaptureMouse)
		{
			const auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
			assert(gw);
			gw->mouseMotion(vec2i(static_cast<int>(x), static_cast<int>(y)));
		}
	}

	auto mouseButton(GLFWwindow* window, const int button, const int action, const int mods) -> void
	{
		const auto& io = ImGui::GetIO();
		if (io.WantCaptureMouse)
		{
			const auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
			assert(gw);
			gw->mouseButton(button, action, mods);
		}
	}

	ImFont* defaultFont;

	auto initializeGui(GLFWwindow* window) -> void
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		auto& io = ImGui::GetIO();
		(void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls

		ImGui::StyleColorsDark();

		const auto monitor = glfwGetPrimaryMonitor();

		float scaleX;
		float scaleY;
		glfwGetMonitorContentScale(monitor, &scaleX, &scaleY);

		const auto dpiScale = scaleX;
		constexpr auto baseFontSize = 18.0f;

		ImFontConfig config;

		config.OversampleH = 1;
		config.OversampleV = 1;
		config.SizePixels = dpiScale * baseFontSize;
		defaultFont = io.Fonts->AddFontDefault(&config);

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
	static auto show_demo_window = true;
	ImGui::ShowDemoWindow(&show_demo_window);
}


auto NanoViewer::showAndRunWithGui() -> void
{
	showAndRunWithGui([]() { return true; });
}

auto NanoViewer::showAndRunWithGui(const std::function<bool()>& keepgoing) -> void
{
	int width, height;
	glfwGetFramebufferSize(handle, &width, &height);
	resize(vec2i(width, height));

	glfwSetFramebufferSizeCallback(handle, reshape);
	glfwSetMouseButtonCallback(handle, ::mouseButton);
	glfwSetKeyCallback(handle, keyboardSpecialKey);
	glfwSetCharCallback(handle, keyboardKey);
	glfwSetCursorPosCallback(handle, ::mouseMotion);

	initializeGui(handle);

	while (!glfwWindowShouldClose(handle) && keepgoing())
	{
		onFrameBegin();
		static double lastCameraUpdate = -1.f;
		if (camera.lastModified != lastCameraUpdate)
		{
			cameraChanged();
			lastCameraUpdate = camera.lastModified;
		}

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::PushFont(defaultFont);
		gui();
		ImGui::PopFont();
		ImGui::EndFrame();

		render();
		draw();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(handle);
		glfwPollEvents();
		onFrameEnd();
	}

	deinitializeGui();
	glfwDestroyWindow(handle);
	glfwTerminate();
}
