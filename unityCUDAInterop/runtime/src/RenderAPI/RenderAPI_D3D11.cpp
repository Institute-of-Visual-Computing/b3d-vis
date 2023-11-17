#include <d3d11.h>

#include "IUnityGraphicsD3D11.h"

#include "PluginLogger.h"

#include "RenderAPI/RenderAPI_D3D11.h"
#include "Texture/Texture_D3D11.h"

using namespace b3d::unity_cuda_interop;
using namespace b3d::unity_cuda_interop::runtime;

RenderAPI_D3D11::RenderAPI_D3D11(const UnityGfxRenderer unityGfxRenderer, IUnityInterfaces* unityInterfaces,
								 PluginLogger* logger)
	: RenderAPI(logger)
{
	unityGraphics_ = unityInterfaces->Get<IUnityGraphicsD3D11>();
	unityGfxRendererType_ = unityGfxRenderer;
	if (unityGraphics_ == nullptr)
	{
		logger_->log("Could not retrieve IUnityGraphicsD3D11 from UnityInterfaces");
	}
}

RenderAPI_D3D11::~RenderAPI_D3D11()
{
	device_ = nullptr;
	unityGraphics_ = nullptr;
}

auto RenderAPI_D3D11::initialize() -> void
{
	if (device_ == nullptr)
	{
		device_ = unityGraphics_->GetDevice();
		if (device_ == nullptr)
		{
			logger_->log("Could not initialize RenderAPI_D3D11");
		}
	}
	else
	{
		logger_->log("RenderAPI_D3D11 was initialized");
	}
}

auto RenderAPI_D3D11::createTexture(void* unityNativeTexturePointer) -> std::unique_ptr<Texture>
{
	return std::make_unique<TextureD3D11>(unityNativeTexturePointer);
}
