#include "App.h"

#include "FastVoxelTraversalRenderer.h"
#include "NullRenderer.h"
#include "Viewer.h"


auto Application::run(const std::vector<Param>& params) -> void
{
	b3d::renderer::registerRenderer<b3d::renderer::NullRenderer>("nullRenderer");
	b3d::renderer::registerRenderer<b3d::renderer::FastVoxelTraversalRenderer>("FastVoxelTraversalRenderer");


	std::cout << b3d::renderer::registry.front().name << std::endl;
	using namespace std::string_literals;
	Viewer viewer("Default Viewer"s);

	viewer.enableFlyMode();
	viewer.enableInspectMode(owl::box3f(owl::vec3f(-1.f), owl::vec3f(+1.f)));
	viewer.showAndRunWithGui();
}
