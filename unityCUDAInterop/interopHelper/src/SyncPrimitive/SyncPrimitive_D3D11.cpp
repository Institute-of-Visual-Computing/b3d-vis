#include "SyncPrimitive/SyncPrimitive_D3D11.h"

#include <d3d11_4.h>

#include "PluginLogger.h"

using namespace b3d::unity_cuda_interop;


SyncPrimitiveD3D11::SyncPrimitiveD3D11(PluginLogger* logger, ID3D11Device* d3d11Device)
	: d3d11Device_(d3d11Device), logger_(logger)
{
	ID3D11Device5* device5;
	auto d3d11Result = d3d11Device_->QueryInterface(__uuidof(ID3D11Device5), reinterpret_cast<void**>(&device5));
	if (d3d11Result != S_OK)
	{
		logger_->log("Could not query d3d11device5 interface. Cant create fence");
		return;
	}

	// D3D11_FENCE_FLAG_SHARED_CROSS_ADAPTER
	d3d11Result = device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, __uuidof(ID3D11Fence),
									reinterpret_cast<void**>(&d3d11Fence_));
	if (d3d11Result != S_OK)
	{
		logger_->log("Could not create d3d11 fence.");
		return;
	}
	
	d3d11Result = d3d11Fence_->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &d3d11SharedFenceHandle_);

	if (d3d11Result != S_OK)
	{
		logger_->log("Could not create Shared handle for d3d11Fence.");
		return;
	}

	isValid_ = true;
	device5->Release();

	/*

		_api->getDevice()->QueryInterface(__uuidof(ID3D11Device5), (void**) & device5);

		device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, __uuidof(ID3D11Fence), (void**)&_fence1);
		device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, __uuidof(ID3D11Fence), (void**)&_fence2);

		ID3D11DeviceContext* ctx;
		_api->getDevice()->GetImmediateContext(&ctx);
		ctx->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&_context4);
	*/
}

SyncPrimitiveD3D11::~SyncPrimitiveD3D11()
{
	if (d3d11Fence_ != nullptr)
	{
		if (cudaSemaphore_ != nullptr)
		{
			auto cudaResult = cudaDestroyExternalSemaphore(cudaSemaphore_);
			if (cudaResult != cudaSuccess)
			{
				logger_->log("Could not destroy external semaphore.");
			}
		}
		d3d11Fence_->Release();
		d3d11Fence_ = nullptr;
	}
	d3d11Device_ = nullptr;
	logger_ = nullptr;
}

auto SyncPrimitiveD3D11::importToCUDA() -> void
{
	if (isValid_)
	{
		cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc{};
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
		externalSemaphoreHandleDesc.handle.win32.handle = d3d11SharedFenceHandle_;
		const auto cudaResult = cudaImportExternalSemaphore(&cudaSemaphore_, &externalSemaphoreHandleDesc);

		if (cudaResult != cudaSuccess)
		{
			logger_->log("Could not import shared fence to cuda.");
			isValid_ = false;
			return;
		}
		CloseHandle(d3d11SharedFenceHandle_);
		d3d11SharedFenceHandle_ = nullptr;
	}
	else
	{
		logger_->log("Can't import fence to CUDA. SyncPrimitive is not valid.");
	}
}
