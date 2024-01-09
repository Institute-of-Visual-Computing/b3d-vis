#include "RenderAPI.h"
#include "IUnityGraphics.h"

#include "RenderAPI/RenderAPI_D3D11.h"

using namespace b3d::unity_cuda_interop;

auto RenderAPI::createRenderAPI(UnityGfxRenderer unityGfxRenderer, IUnityInterfaces* unityInterfaces,
								PluginLogger* logger) -> std::unique_ptr<RenderAPI>
{
	switch (unityGfxRenderer)
	{
	case kUnityGfxRendererD3D11:
		return std::make_unique<RenderAPI_D3D11>(unityGfxRenderer, unityInterfaces, logger);
	default:
		break;
	}
	return nullptr;
}

RenderAPI::~RenderAPI() = default;
