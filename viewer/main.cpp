#include "NanoViewer.h"
#include <imgui.h>
#include <owl/owl.h>

struct Viewer : public NanoViewer
{
	Viewer();

	void render() override;
	void resize(const owl::vec2i& newSize) override;

	void cameraChanged() override;

	void gui() override;
};

void Viewer::gui()
{
	ImGui::ShowDemoWindow();
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
}

void Viewer::render()
{
}

int main(int argc, char** argv)
{

	Viewer viewer;


	viewer.enableFlyMode();
	viewer.enableInspectMode(owl::box3f(owl::vec3f(-1.f), owl::vec3f(+1.f)));
	viewer.showAndRunWithGui();
}
