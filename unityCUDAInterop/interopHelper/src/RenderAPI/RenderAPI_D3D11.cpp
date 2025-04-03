#include <d3d11.h>
#include <dxgi.h>

#include "IUnityGraphicsD3D11.h"

#include "PluginLogger.h"

#include "RenderAPI/RenderAPI_D3D11.h"
#include "RenderingContext/RenderingContext_D3D11.h"
#include "SyncPrimitive/SyncPrimitive_D3D11.h"
#include "Texture/Texture_D3D11.h"

using namespace b3d::unity_cuda_interop;

RenderAPI_D3D11::RenderAPI_D3D11(const UnityGfxRenderer unityGfxRenderer, IUnityInterfaces* unityInterfaces,
								 PluginLogger* logger)
	: RenderAPI(logger)
{
	unityInterfaces_ = unityInterfaces;
	unityGraphics_ = unityInterfaces->Get<IUnityGraphicsD3D11>();
	unityGfxRendererType_ = unityGfxRenderer;

	if (unityGraphics_ == nullptr)
	{
		logger_->log("Could not retrieve IUnityGraphicsD3D11 from UnityInterfaces");
	}
	else
	{
		logger_->log("Plugin RenderAPI_D3D11 created");
	}
}

RenderAPI_D3D11::~RenderAPI_D3D11()
{
	device_ = nullptr;
	unityGraphics_ = nullptr;
}

auto RenderAPI_D3D11::initialize() -> void
{
	//if (device_ == nullptr)
	//{
		device_ = unityGraphics_->GetDevice();

		if (device_ == nullptr)
		{
			logger_->log("Could not initialize RenderAPI_D3D11");
			return;
		}
		logger_->log("RenderAPI_D3D11 initialized");
		//}
		// else
		//{
		logger_->log("RenderAPI_D3D11 was already initialized");
		//}
	getCudaDevice();
}

auto RenderAPI_D3D11::createTexture(void* unityNativeTexturePointer) -> std::unique_ptr<Texture>
{
	return std::make_unique<TextureD3D11>(unityNativeTexturePointer);
}

auto RenderAPI_D3D11::createSynchronizationPrimitive() -> std::unique_ptr<SyncPrimitive>
{
	return std::make_unique<SyncPrimitiveD3D11>(logger_, device_);
}
auto RenderAPI_D3D11::createRenderingContext() -> std::unique_ptr<RenderingContext>
{
	return std::make_unique<RenderingContext_D3D11>(logger_, device_);
}


auto RenderAPI_D3D11::getCudaDevice() -> void
{
	IDXGIDevice* dxgiDevice{ nullptr };

	auto result = device_->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice));
	if (result != S_OK)
	{
		logger_->log("Could not query dxgiDevice");
	}

	IDXGIAdapter* dxgiAdapter{ nullptr };
	result = dxgiDevice->GetAdapter(&dxgiAdapter);
	if (result != S_OK)
	{
		logger_->log("Could not get Adapter from dxgiDevice");
	}

	DXGI_ADAPTER_DESC dxgiAdapterDesc;
	result = dxgiAdapter->GetDesc(&dxgiAdapterDesc);
	if (result != S_OK)
	{
		logger_->log("Could not get Description from dxgiAdapter");
	}

	auto d3d11Luid = dxgiAdapterDesc.AdapterLuid;

	int cudaDeviceCount;
	auto cudaResult = cudaGetDeviceCount(&cudaDeviceCount);
	if (cudaResult != cudaSuccess)
	{
		logger_->log("Could not get cuda device count.");
	}

	for (auto cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++)
	{
		cudaDeviceProp deviceProp;
		cudaResult = cudaGetDeviceProperties(&deviceProp, cudaDevice);
		if (cudaResult != cudaSuccess)
		{
			logger_->log("Could not get cuda device count.");
		}

		auto cudaLuid = deviceProp.luid;

		if (!memcmp(&d3d11Luid.LowPart, cudaLuid, sizeof(d3d11Luid.LowPart)) &&
			!memcmp(&d3d11Luid.HighPart, cudaLuid + sizeof(d3d11Luid.LowPart), sizeof(d3d11Luid.HighPart)))
		{

			cudaUUID_ = deviceProp.uuid;
			return;
		}
	}
	logger_->log("Could not get cudaUUID from d3d11 device.");
}
