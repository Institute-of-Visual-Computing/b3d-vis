#pragma once

#include "Texture.h"

struct ID3D11Texture2D;

namespace b3d::unity_cuda_interop
{
	class TextureD3D11 final : public Texture
	{
	public:
		TextureD3D11(void* unityNativeTexturePointer);

		~TextureD3D11() override;

		auto registerCUDA(unsigned registerFlags = cudaGraphicsRegisterFlagsNone,
						  unsigned mapFlags = cudaGraphicsMapFlagsNone) -> void override;


	private:
		ID3D11Texture2D* d3d11GraphicsResource_{ nullptr };
	};
} // namespace b3d::unity_cuda_interop
