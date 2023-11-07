#include "owl/owl.h"
#include "NanoViewer.h"

#include <imgui.h>


struct Viewer : public NanoViewer
{
	Viewer();

	void render() override;
	void resize(const vec2i& newSize) override;

	void cameraChanged() override;

	void gui() override;


void Viewer::gui()
{
	ImGui::ShowDemoWindow();
}

void Viewer::resize(const vec2i& newSize)
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
	auto stream = viewer.getCudaStream();
	handle.buffer().deviceUpload(stream, true);

	auto* deviceGrid = handle.deviceGrid<float>();

	viewer.camera.setOrientation(init_lookFrom, init_lookAt, init_lookUp, owl::viewer::toDegrees(acosf(init_cosFovy)));

	viewer.enableFlyMode();
	viewer.enableInspectMode(owl::box3f(vec3f(-1.f), vec3f(+1.f)));
	viewer.showAndRunWithGui();
}
