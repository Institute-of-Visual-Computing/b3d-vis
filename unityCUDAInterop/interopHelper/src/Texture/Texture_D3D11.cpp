#include <cassert>
#include <cuda_d3d11_interop.h>

#include "Texture/Texture_D3D11.h"

using namespace b3d::unity_cuda_interop;

TextureD3D11::TextureD3D11(void* unityNativeTexturePointer) : Texture(unityNativeTexturePointer)
{
	assert(unityNativeTexturePointer != nullptr);

	d3d11GraphicsResource_ = static_cast<ID3D11Texture2D*>(unityNativeTexturePointer);

	D3D11_TEXTURE2D_DESC texDesc;
	d3d11GraphicsResource_->GetDesc(&texDesc);

	width_ = texDesc.Width;
	height_ = texDesc.Height;
	depth_ = texDesc.ArraySize;

	switch (texDesc.Format)
	{

	case DXGI_FORMAT_R32G32B32_TYPELESS:
	case DXGI_FORMAT_R16G16B16A16_TYPELESS:
	case DXGI_FORMAT_R32G32_TYPELESS:
	case DXGI_FORMAT_R32G8X24_TYPELESS:
	case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
	case DXGI_FORMAT_R10G10B10A2_TYPELESS:
	case DXGI_FORMAT_R8G8B8A8_TYPELESS:
	case DXGI_FORMAT_R16G16_TYPELESS:
	case DXGI_FORMAT_R32_TYPELESS:
	case DXGI_FORMAT_R24G8_TYPELESS:
	case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
	case DXGI_FORMAT_R8G8_TYPELESS:
	case DXGI_FORMAT_R16_TYPELESS:
	case DXGI_FORMAT_R8_TYPELESS:
	case DXGI_FORMAT_BC1_TYPELESS:
	case DXGI_FORMAT_BC2_TYPELESS:
	case DXGI_FORMAT_BC3_TYPELESS:
	case DXGI_FORMAT_BC4_TYPELESS:
	case DXGI_FORMAT_BC5_TYPELESS:
	case DXGI_FORMAT_B8G8R8A8_TYPELESS:
	case DXGI_FORMAT_B8G8R8X8_TYPELESS:
	case DXGI_FORMAT_BC6H_TYPELESS:
	case DXGI_FORMAT_BC7_TYPELESS:
		isValid_ = false;
		break;
	default:
		isValid_ = true;
	}
}

TextureD3D11::~TextureD3D11()
{
	d3d11GraphicsResource_ = nullptr;
}

auto TextureD3D11::registerCUDA() -> void
{
	assert(d3d11GraphicsResource_ != nullptr);
	// TODO: Error handling
	cudaGraphicsD3D11RegisterResource(&cudaGraphicsResource_, d3d11GraphicsResource_, cudaGraphicsRegisterFlagsNone);
}
