#include "NanoViewer.h"
#include <NullRenderer.h>
#include <RendererBase.h>
#include <imgui.h>
#include <owl/owl.h>

struct Viewer : public NanoViewer
{
	Viewer();

	void render() override;
	void resize(const owl::vec2i& newSize) override;

	void cameraChanged() override;

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
};

void Viewer::gui()
{
	ImGui::ShowDemoWindow();
	currentRenderer->gui();
}

void Viewer::resize(const owl::vec2i& newSize)
{
	OWLViewer::resize(newSize);
	cameraChanged();
}

void Viewer::cameraChanged()
{
}

Viewer::Viewer()
{
	selectRenderer(0);
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

	Viewer viewer;

	viewer.enableFlyMode();
	viewer.enableInspectMode(owl::box3f(owl::vec3f(-1.f), owl::vec3f(+1.f)));
	viewer.showAndRunWithGui();
}
