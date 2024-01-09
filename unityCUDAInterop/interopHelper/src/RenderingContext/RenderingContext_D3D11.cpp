#include "RenderingContext/RenderingContext_D3D11.h"

#include "PluginLogger.h"

b3d::unity_cuda_interop::RenderingContext_D3D11::RenderingContext_D3D11(PluginLogger* logger,
																				 ID3D11Device* d3d11Device)
	: d3d11Device_(d3d11Device), logger_(logger)
{

}

auto b3d::unity_cuda_interop::RenderingContext_D3D11::signal(SyncPrimitive* syncPrimitive, unsigned value)
	-> void
{
	const auto syncPrimitiveD3D11 = static_cast<SyncPrimitiveD3D11*>(syncPrimitive);

	ID3D11DeviceContext* deviceContext;
	ID3D11DeviceContext4* deviceContext4;
	d3d11Device_->GetImmediateContext(&deviceContext);
	deviceContext->QueryInterface(__uuidof(ID3D11DeviceContext4), reinterpret_cast<void**>(&deviceContext4));
	deviceContext4->Signal(syncPrimitiveD3D11->getD3D11Fence(), value);
	ReleaseActCtx(deviceContext);
}

auto b3d::unity_cuda_interop::RenderingContext_D3D11::wait(SyncPrimitive* syncPrimitive, unsigned value) -> void
{
	const auto syncPrimitiveD3D11 = static_cast<SyncPrimitiveD3D11*>(syncPrimitive);

	ID3D11DeviceContext* deviceContext;
	ID3D11DeviceContext4* deviceContext4;
	d3d11Device_->GetImmediateContext(&deviceContext);
	deviceContext->QueryInterface(__uuidof(ID3D11DeviceContext4), reinterpret_cast<void**>(&deviceContext4));
	deviceContext4->Wait(syncPrimitiveD3D11->getD3D11Fence(), value);
	ReleaseActCtx(deviceContext);
}
