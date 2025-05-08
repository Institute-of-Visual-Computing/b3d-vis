#pragma once
#include <memory>

#include "RenderingContext.h"
#include "Texture.h"
#include "SyncPrimitive.h"

#pragma warning (push, 0)
enum UnityGfxRenderer : int;
#pragma warning (pop)
struct IUnityInterfaces;


namespace b3d::unity_cuda_interop
{
	class PluginLogger;

		class RenderAPI
		{
		public:
			static auto createRenderAPI(UnityGfxRenderer unityGfxRenderer, IUnityInterfaces* unityInterfaces,
										PluginLogger* logger) -> std::unique_ptr<RenderAPI>;

			RenderAPI() = delete;

			virtual ~RenderAPI();

			virtual auto initialize() -> void = 0;

			virtual auto createTexture(void* unityNativeTexturePointer) -> std::unique_ptr<Texture> = 0;

			virtual auto createSynchronizationPrimitive() -> std::unique_ptr<SyncPrimitive> = 0;

			virtual auto createRenderingContext() -> std::unique_ptr<RenderingContext> = 0;

			auto getCudaUUID() const -> cudaUUID_t
			{
				return cudaUUID_;
			}

			auto getUnityInterfaces() -> IUnityInterfaces*
			{
				return unityInterfaces_;
			}

		protected:
			RenderAPI(PluginLogger* logger) : logger_(logger)
			{
			}

			virtual auto getCudaDevice() -> void = 0;

			cudaUUID_t cudaUUID_{};
			PluginLogger* logger_;
			UnityGfxRenderer unityGfxRendererType_;
			IUnityInterfaces* unityInterfaces_;
		};
} // namespace b3d::unity_cuda_interop
