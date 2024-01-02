#include "NanoViewer.h"
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

auto NanoViewer::draw1() -> void
{
      //resourceSharingSuccessful = false;
      if (resourceSharingSuccessful) {
		/* 
        OWL_CUDA_CHECK(cudaGraphicsMapResources(1, &cuDisplayTexture));

        cudaArray_t array;
        OWL_CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, cuDisplayTexture, 0, 0));
        {
          cudaMemcpy2DToArray(array,
                              0,
                              0,
                              reinterpret_cast<const void *>(fbPointer),
                              fbSize.x * sizeof(uint32_t),
                              fbSize.x * sizeof(uint32_t),
                              fbSize.y,
                              cudaMemcpyDeviceToDevice);

        }
        */
      } else {
        glBindTexture(GL_TEXTURE_2D, fbTexture);
        glEnable(GL_TEXTURE_2D);
        glTexSubImage2D(GL_TEXTURE_2D,0,
                                 0,0,
                                 fbSize.x, fbSize.y,
                                 GL_RGBA, GL_UNSIGNED_BYTE, fbPointer);
      }

      glDisable(GL_LIGHTING);
      glColor3f(1, 1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      glDisable(GL_DEPTH_TEST);

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);

        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);

        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
      }
      glEnd();
      if (resourceSharingSuccessful) {
        // OWL_CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuDisplayTexture));
      }
    }

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
	glfwMakeContextCurrent(handle);
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
		draw1();

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
