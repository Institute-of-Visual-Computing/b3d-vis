#pragma once

#include "RenderAPI.h"

struct IUnityGraphicsD3D11;
struct ID3D11Device;
// ReSharper disable once CppInconsistentNaming
enum class kUnityGfxRendererD3D11;

using namespace b3d::unity_cuda_interop;
using namespace b3d::unity_cuda_interop::runtime;

// ReSharper disable once CppInconsistentNaming
class RenderAPI_D3D11 final : public RenderAPI
{
	public:
		RenderAPI_D3D11(UnityGfxRenderer unityGfxRenderer, IUnityInterfaces* unityInterfaces,
						PluginLogger* logger);

		~RenderAPI_D3D11() override;
		auto initialize() -> void override;

		auto createTexture(void* unityNativeTexturePointer) -> std::unique_ptr<Texture> override;

		auto createSynchronizationPrimitive() -> std::unique_ptr<SyncPrimitive> override;

		auto createRenderingContext() -> std::unique_ptr<RenderingContext> override;

		auto getD3D11Device() const -> ID3D11Device*
		{
			return device_;
		};

protected:
		auto getCudaDevice() -> void override;

private:
	IUnityGraphicsD3D11 *unityGraphics_ { nullptr };
	ID3D11Device *device_ { nullptr };
};
