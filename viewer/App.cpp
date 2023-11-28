#include "App.h"

// #include "FastVoxelTraversalRenderer.h"
#include "CudaSurfaceObjectWriteTestRenderer.h"
#include "NanoRenderer.h"
#include "NullRenderer.h"
#include "Viewer.h"

using namespace b3d::renderer;;

auto Application::run() -> void
{
	registerRenderer<NullRenderer>("nullRenderer");
	// registerRenderer<FastVoxelTraversalRenderer>("FastVoxelTraversalRenderer");
	registerRenderer<CudaSurfaceObjectWriteTestRenderer>("CudaSurfaceObjectWriteTestRenderer");
	registerRenderer<NanoRenderer>("NanoRenderer");

	std::cout << registry.front().name << std::endl;
	using namespace std::string_literals;
	Viewer viewer("Default Viewer"s);

	viewer.enableFlyMode();
	viewer.enableInspectMode(owl::box3f(owl::vec3f(-1.f), owl::vec3f(+1.f)));
	viewer.showAndRunWithGui();
}
auto Application::initialization(const std::vector<Param>& vector) -> void
{
}
