#pragma once

#include "RenderAPI.h"

struct IUnityGraphicsD3D11;
struct ID3D11Device;
enum class kUnityGfxRendererD3D11;
class RenderAPI_D3D11 : public RenderAPI
{
	public:
		RenderAPI_D3D11(UnityGfxRenderer unityGfxRenderer, IUnityInterfaces* unityInterfaces, PluginLogger* logger);

		~RenderAPI_D3D11();
		void initialize() override;

		std::unique_ptr<Texture> createTexture(void* unityNativeTexturePointer) override;

private:
	IUnityGraphicsD3D11 *unityGraphics_ { nullptr };
	ID3D11Device *device_ { nullptr };
	
};
