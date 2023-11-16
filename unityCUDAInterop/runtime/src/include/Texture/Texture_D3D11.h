#pragma once

#include "Texture.h"

struct ID3D11Texture2D;

class TextureD3D11 final : public Texture
{
public:
	TextureD3D11(void* unityNativeTexturePointer);

	~TextureD3D11() override;

	void registerCUDA() override;
	
protected:

private:
	ID3D11Texture2D* d3d11GraphicsResource_{ nullptr };
};