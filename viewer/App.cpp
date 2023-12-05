#include "App.h"

// #include "FastVoxelTraversalRenderer.h"
#include "CudaSurfaceObjectWriteTestRenderer.h"
#include "NanoRenderer.h"
#include "NullRenderer.h"
#include "SimpleTrianglesRenderer.h"
#include "Viewer.h"

using namespace b3d::renderer;;

namespace 
{
	int rendererIndex = 0;
}

auto Application::run() -> void
{
	std::cout << registry.front().name << std::endl;
	using namespace std::string_literals;
	auto viewer = Viewer{"Default Viewer"s, 1980, 1080, rendererIndex };
	viewer.enableFlyMode();
	viewer.enableInspectMode(owl::box3f(owl::vec3f(-1.f), owl::vec3f(+1.f)));
	viewer.showAndRunWithGui();
}
auto Application::initialization(const std::vector<Param>& vector) -> void
{
	registerRenderer<NullRenderer>("nullRenderer");
	registerRenderer<NanoRenderer>("NanoRenderer");
	registerRenderer<SimpleTrianglesRenderer>("SimpleTrianglesRenderer");
	// registerRenderer<FastVoxelTraversalRenderer>("FastVoxelTraversalRenderer");
	registerRenderer<CudaSurfaceObjectWriteTestRenderer>("CudaSurfaceObjectWriteTestRenderer");

	auto found = std::find_if(vector.begin(), vector.end(), [](const Param& param)
	{
		if(param.value == "--renderer")
		{
			return true;
		}
	  return  false;
	});

	if(found != vector.end())
	{
		assert((found++) != vector.end());

		const auto value = *found;

		const auto index = getRendererIndex(value.value);

		if(index != -1)
		{
			rendererIndex = index;
		}
	}

}
