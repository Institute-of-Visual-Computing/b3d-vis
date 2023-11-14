#include "NanoViewer.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
using namespace owl;
using namespace owl::viewer;

namespace
{
	void reshape(GLFWwindow* window, int width, int height)
	{
		auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
		assert(gw);
		gw->resize(vec2i(width, height));
	}

	void keyboardKey(GLFWwindow* window, unsigned int key)
	{
		auto& io = ImGui::GetIO();
		if(io.WantCaptureKeyboard)
		{
			auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
			assert(gw);
			gw->key(key, gw->getMousePos());
		}
	}

	void keyboardSpecialKey(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
		assert(gw);
		if(action == GLFW_PRESS)
		{
			gw->special(key, mods, gw->getMousePos());
		}
	}

	void mouseMotion(GLFWwindow* window, double x, double y)
	{
		auto& io = ImGui::GetIO();
		if(io.WantCaptureMouse)
		{
			auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
			assert(gw);
			gw->mouseMotion(vec2i((int)x, (int)y));
		}
	}

	void mouseButton(GLFWwindow* window, int button, int action, int mods)
	{
		auto& io = ImGui::GetIO();
		if(io.WantCaptureMouse)
		{
			auto gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
			assert(gw);
			gw->mouseButton(button, action, mods);
		}
	}

	ImFont* defaultFont;
} // namespace

void NanoViewer::gui()
{
	static bool show_demo_window = true;
	ImGui::ShowDemoWindow(&show_demo_window);
}

void NanoViewer::initializeGui()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

	ImGui::StyleColorsDark();

	auto monitor = glfwGetPrimaryMonitor();

	float scaleX;
	float scaleY;
	glfwGetMonitorContentScale(monitor, &scaleX, &scaleY);

	const auto dpiScale = scaleX;
	float baseFontSize = 18.0f;
	
	ImFontConfig config;

	config.OversampleH = 1;
	config.OversampleV = 1;
	config.SizePixels = dpiScale * baseFontSize;
	defaultFont = io.Fonts->AddFontDefault(&config);

	ImGui_ImplGlfw_InitForOpenGL(handle, true);
	ImGui_ImplOpenGL3_Init();
}

void NanoViewer::deinitializeGui()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();


	ImGui::DestroyContext();
}

void NanoViewer::showAndRunWithGui()
{
	showAndRunWithGui([]() { return true; });
}

void NanoViewer::showAndRunWithGui(std::function<bool()> keepgoing)
{
	int width, height;
	glfwGetFramebufferSize(handle, &width, &height);
	resize(vec2i(width, height));

	glfwSetFramebufferSizeCallback(handle, reshape);
	glfwSetMouseButtonCallback(handle, ::mouseButton);
	glfwSetKeyCallback(handle, keyboardSpecialKey);
	glfwSetCharCallback(handle, keyboardKey);
	glfwSetCursorPosCallback(handle, ::mouseMotion);

	initializeGui();
	
	while(!glfwWindowShouldClose(handle) && keepgoing())
	{
		onFrameBegin();
		static double lastCameraUpdate = -1.f;
		if(camera.lastModified != lastCameraUpdate)
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
