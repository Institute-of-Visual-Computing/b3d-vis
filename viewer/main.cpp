#define GLFW_INCLUDE_GLEXT
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "NanoViewer.h"
#include <NullRenderer.h>
#include <RendererBase.h>
#include <imgui.h>

#include <owl/owl.h>
#include <driver_types.h>


enum class RenderMode
{
	mono,
	stereo 
};

enum class SemaphoreState
{
	signaled,
	unsignaled
};

struct TestSemaphore final
{
	void signal()
	{
		ReleaseSemaphore(handle, 1, nullptr);
	}
	void wait()
	{
		WaitForSingleObject(handle, INFINITE);
	}

	TestSemaphore(SemaphoreState initialState = SemaphoreState::unsignaled)
	{
		handle = CreateSemaphore(NULL, initialState == SemaphoreState::signaled ? 1 : 0, 1, NULL);
	}

	~TestSemaphore()
	{
		CloseHandle(handle);
	}

	HANDLE handle;
};

struct RendererInitializationInfo
{
	// on stereo we expect that those resources are of a array type in native API
	cudaGraphicsResource_t colorRT;
	cudaGraphicsResource_t minMaxRT;
	RenderMode mode{ RenderMode::mono };

	cudaExternalSemaphore_t waitSemaphore;
	cudaExternalSemaphore_t signalSemaphore;
};


struct Viewer : public NanoViewer
{
	Viewer(const std::string& title = "Sample Viewer", const int initWindowWidth = 1980,
	       const int initWindowHeight = 1080);

	void render() override;
	void resize(const owl::vec2i& newSize) override;

	void cameraChanged() override;

	void onFrameBegin() override;

	void gui() override;

  private:
	void selectRenderer(const std::uint32_t index)
	{
		if(selectedRendererIndex == index)
			return;
		if(currentRenderer)
		{
			currentRenderer->deinitialize();
		}

		assert(index < b3d::registry.size());
		selectedRendererIndex = index;
		currentRenderer = b3d::registry[selectedRendererIndex].rendererInstance;

		currentRenderer->initialize();
	}
	std::shared_ptr<b3d::RendererBase> currentRenderer{ nullptr };
	std::int32_t selectedRendererIndex{ -1 };

	std::int32_t newSelectedRendererIndex{ -1 };

	std::vector<std::string> registeredRendererNames{};

	TestSemaphore waitSemaphore{};
	TestSemaphore signalSemaphore{};

	GLuint colorTexture;
	GLuint minMaxTexture;

	RendererInitializationInfo rendererInfo{};

	RenderMode mode{ RenderMode::mono };
};

void Viewer::onFrameBegin()
{
	if (newSelectedRendererIndex != selectedRendererIndex)
	{
		selectRenderer(newSelectedRendererIndex);
	}
}

void Viewer::gui()
{
	ImGui::ShowDemoWindow();
	currentRenderer->gui();

	ImGui::Begin("Renderer Selection");


	const auto preview = registeredRendererNames[selectedRendererIndex];

	if(ImGui::BeginCombo("Renderer", preview.c_str()))
	{
		for(int n = 0; n < registeredRendererNames.size(); n++)
		{
			const bool isSelected = (selectedRendererIndex == n);
			if(ImGui::Selectable(registeredRendererNames[n].c_str(), isSelected))
				newSelectedRendererIndex = n;

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

Viewer::Viewer(const std::string& title, const int initWindowWidth,
               const int initWindowHeight)
    : NanoViewer(title, initWindowWidth, initWindowHeight)
{
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	gladLoadGL();

	auto semaphoreHandleInfo = cudaExternalSemaphoreHandleDesc{};
	semaphoreHandleInfo.flags = 0;
	semaphoreHandleInfo.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	semaphoreHandleInfo.handle.win32.handle = waitSemaphore.handle;
	cudaImportExternalSemaphore(&rendererInfo.waitSemaphore, &semaphoreHandleInfo);

	semaphoreHandleInfo.handle.win32.handle = signalSemaphore.handle;
	cudaImportExternalSemaphore(&rendererInfo.signalSemaphore, &semaphoreHandleInfo);

	glGenTextures(1, &colorTexture);
	glBindTexture(GL_TEXTURE_2D_ARRAY, colorTexture);
	glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, 64, 64, 2);

	glGenTextures(1, &minMaxTexture);
	glBindTexture(GL_TEXTURE_2D_ARRAY, minMaxTexture);
	glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RG32F, 64, 64, 2);

	cudaGraphicsGLRegisterImage(&rendererInfo.colorRT, colorTexture,
	                            mode == RenderMode::mono ? GL_TEXTURE_2D : GL_TEXTURE_2D_ARRAY,
	                            cudaGraphicsRegisterFlagsWriteDiscard);

	cudaGraphicsGLRegisterImage(&rendererInfo.minMaxRT, colorTexture,
	                            mode == RenderMode::mono ? GL_TEXTURE_2D : GL_TEXTURE_2D_ARRAY,
	                            cudaGraphicsRegisterFlagsWriteDiscard);

	rendererInfo.mode = mode;

	//NOTE: rendererInfo will be feeded into renderer initialization


	selectRenderer(0);
	newSelectedRendererIndex = selectedRendererIndex;

	for (int i = 0; i < b3d::registry.size(); i++)
	{
		registeredRendererNames.push_back(b3d::registry[i].name);
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

	currentRenderer->render(view);

}

int main(int argc, char** argv)
{
	

	b3d::registerRenderer<b3d::NullRenderer>("nullRenderer");

	std::cout << b3d::registry.front().name << std::endl;
	using namespace std::string_literals;
	Viewer viewer("Default Viewer"s);

	viewer.enableFlyMode();
	viewer.enableInspectMode(owl::box3f(owl::vec3f(-1.f), owl::vec3f(+1.f)));
	viewer.showAndRunWithGui();
}
