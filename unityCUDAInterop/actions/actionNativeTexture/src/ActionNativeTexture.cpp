#include "Action.h"

#include "PluginLogger.h"
#include "RenderAPI.h"

#include <d3d11.h>
#include <dxgi.h>
#include "IUnityGraphicsD3D11.h"

#include "RenderData.h"

#include <windows.h>
#include <stdio.h>
#include <comdef.h>


#include "TextureFillerSyncSample.h"

// This line is crucial and must stay. Should be one of the last include. But in any case after the include of Action.
#include <cuda_d3d11_interop.h>
#include <d3d11_4.h>
#include <format>

#include "create_action.h"
#include "Logging.h"
#include "NullDebugDrawList.h"
#include "NullGizmoHelper.h"

using namespace b3d::unity_cuda_interop;

enum class NativeTextureEventTypes : int
{
	initializeEvent = 0,
	setTexturesEvent,
	beforeForwardAlpha,
};

struct UnityActionNativeTextureData
{
	UnityExtent textureDimensions{};
	ID3D11Resource* nativeTexturePointer = nullptr;
};

static const b3d::renderer::SchemaData unityActionNativeTextureDataSchema{
	{
		SCHEMA_ENTRY("textureDimensions", textureDimensions, UnityActionNativeTextureData),
		SCHEMA_ENTRY("nativeTexturePointer", nativeTexturePointer, UnityActionNativeTextureData),

	},
	sizeof(UnityActionNativeTextureData)
};

class ActionNativeTexture final : public Action
{
public:
	ActionNativeTexture();
	auto initialize(void* data) -> void override;
	auto teardown() -> void override;

protected:
	auto customRenderEvent(int eventId, void* data) -> void override;
	ID3D11Texture2D* myTexture_;
	std::unique_ptr<Texture> myWrappedTexture_;

	
	// std::unique_ptr<RenderingContext> renderingContext_;
	// std::unique_ptr<SyncPrimitive> waitPrimitive_;
	// std::unique_ptr<SyncPrimitive> signalPrimitive_;

	std::unique_ptr<b3d::renderer::TextureFillerSyncSample> renderer_;
	std::atomic_uint64_t currFenceValue = 0;
	std::atomic_flag resourceCreated = ATOMIC_FLAG_INIT;
	
};

ActionNativeTexture::ActionNativeTexture()
{
	renderer_ = std::make_unique<b3d::renderer::TextureFillerSyncSample>();
}

auto ActionNativeTexture::initialize(void* data) -> void
{
	if (data == nullptr)
	{
		return;
	}

	


	// cudaSetDevice();
	

	b3d::renderer::RenderingDataBuffer nativeTextureDataBuffer{ unityActionNativeTextureDataSchema, 1, data };
	auto device = renderAPI_->getUnityInterfaces()->Get<IUnityGraphicsD3D11>()->GetDevice();

	ID3D11DeviceContext* immediateContext;
	device->GetImmediateContext(&immediateContext);
	ID3D11Multithread* id3d11multiThread;
	auto result = immediateContext->QueryInterface(__uuidof(ID3D11Multithread), reinterpret_cast<void**>(&id3d11multiThread));
	if (result != S_OK)
	{
		logger_->log("Could not query ID3D11Multithread");
	}
	id3d11multiThread->SetMultithreadProtected(true);


	IDXGIDevice* dxgiDevice{ nullptr };

	result = device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice));
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

	int cudaDevice;
	cudaD3D11GetDevice(&cudaDevice, dxgiAdapter);

	cudaSetDevice(cudaDevice);
	cudaDeviceProp cudaDevProps;
	cudaGetDeviceProperties(&cudaDevProps, cudaDevice);


	auto id = std::this_thread::get_id();
	std::cout << "Thread ID: " << id << "\n";
	auto lala = device->GetFeatureLevel();
	auto extent = nativeTextureDataBuffer.get<UnityExtent>("textureDimensions");
	if (extent->width + extent->height + extent->depth == 0)
	{
		return;
	}

	D3D11_TEXTURE2D_DESC desc;
	desc.Width = 256;
	desc.Height = 256;
	desc.MipLevels = desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

	ID3D11Texture2D* pTexture = NULL;
	HRESULT hresult = device->CreateTexture2D(&desc, NULL, &pTexture);
	logger_->log(std::format("{}", sizeof(UnityActionNativeTextureData)).c_str());
	if (SUCCEEDED(hresult))
	{
		myTexture_ = pTexture;

		auto txtPtr = nativeTextureDataBuffer.get<ID3D11Resource*>("nativeTexturePointer");
		*txtPtr = myTexture_;
		logger_->log("Created");
		myWrappedTexture_ = renderAPI_->createTexture(myTexture_);
		myWrappedTexture_->registerCUDA();


		// Get Sync Primitives
		// waitPrimitive_ = renderAPI_->createSynchronizationPrimitive();
		// signalPrimitive_ = renderAPI_->createSynchronizationPrimitive();
		// waitPrimitive_->importToCUDA();
		// signalPrimitive_->importToCUDA();

		// renderingContext_ = renderAPI_->createRenderingContext();

		// renderingDataWrapper_.data.synchronization.waitSemaphore = waitPrimitive_->getCudaSemaphore();
		// renderingDataWrapper_.data.synchronization.signalSemaphore = signalPrimitive_->getCudaSemaphore();

		renderingDataWrapper_.data.rendererInitializationInfo.deviceUuid = cudaDevProps.uuid; //renderAPI_->getCudaUUID();

		renderer_->initialize(
			&renderingDataWrapper_.buffer,
			b3d::renderer::DebugInitializationInfo{ std::make_shared<b3d::renderer::NullDebugDrawList>(),
													std::make_shared<b3d::renderer::NullGizmoHelper>() });
		isInitialized_ = true;
		cudaDeviceSynchronize();
		resourceCreated.test_and_set();
		resourceCreated.notify_all();
	}
	else
	{
		_com_error err(hresult);
		LPCTSTR errMsg = err.ErrorMessage();
		logger_->log(errMsg);
	}
	

	
}

auto ActionNativeTexture::teardown() -> void
{
	auto device = renderAPI_->getUnityInterfaces()->Get<IUnityGraphicsD3D11>()->GetDevice();
	isInitialized_ = false;

	renderer_->deinitialize();
	cudaDeviceSynchronize();
	renderer_.reset();
	myTexture_->Release();
	// waitPrimitive_.reset();
	// signalPrimitive_.reset();
	// renderingContext_.reset();
}

auto ActionNativeTexture::customRenderEvent(int eventId, void* data) -> void
{
	if (eventId == static_cast<int>(NativeTextureEventTypes::initializeEvent) && !isInitialized_)
	{
	}
	if (eventId == static_cast<int>(NativeTextureEventTypes::beforeForwardAlpha))
	{
		if (resourceCreated.test() && isInitialized_)
		{
			

			auto device = renderAPI_->getUnityInterfaces()->Get<IUnityGraphicsD3D11>()->GetDevice();
			ID3D11DeviceContext* immediateContext;
			device->GetImmediateContext(&immediateContext);
			ID3D11Multithread* id3d11multiThread;
			auto result = immediateContext->QueryInterface(__uuidof(ID3D11Multithread),
														   reinterpret_cast<void**>(&id3d11multiThread));
			if (result != S_OK)
			{
				logger_->log("Could not query ID3D11Multithread");
			}

			IDXGIDevice* dxgiDevice{ nullptr };

			result = device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice));
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

			int cudaDevice;
			cudaD3D11GetDevice(&cudaDevice, dxgiAdapter);

			cudaSetDevice(cudaDevice);
			cudaDeviceProp cudaDevProps;
			cudaGetDeviceProperties(&cudaDevProps, cudaDevice);

			logger_->log("Render");

			renderingDataWrapper_.data.renderTargets.colorRt = {
				.target = myWrappedTexture_->getCudaGraphicsResource(),
				.extent = { static_cast<uint32_t>(myWrappedTexture_->getWidth()),
							static_cast<uint32_t>(myWrappedTexture_->getHeight()), 1 }
			};

			currFenceValue += 1;
			renderingDataWrapper_.data.synchronization.fenceValue = currFenceValue;


			renderer_->render();
		}
		else
		{
			logger_->log("Test failed");
			initialize(data);
		}
	}
}

// This line is crucial and must stay. Replace type with your newly created action type.
EXTERN_CREATE_ACTION(ActionNativeTexture)
