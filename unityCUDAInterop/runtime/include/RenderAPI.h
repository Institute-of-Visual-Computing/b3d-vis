#pragma once
#include <memory>

#include "Texture.h"

enum UnityGfxRenderer : int;
struct IUnityInterfaces;

namespace b3d::unity_cuda_interop
{
	class PluginLogger;

	namespace runtime
	{
		class RenderAPI
		{
		public:
			static auto createRenderAPI(UnityGfxRenderer unityGfxRenderer, IUnityInterfaces* unityInterfaces,
										PluginLogger* logger) -> std::unique_ptr<RenderAPI>;

			RenderAPI() = delete;

			virtual ~RenderAPI();

			// Use this only on the renderThread
			virtual auto initialize() -> void = 0;

			virtual auto createTexture(void* unityNativeTexturePointer) -> std::unique_ptr<Texture> = 0;

		protected:
			RenderAPI(PluginLogger* logger) : logger_(logger)
			{
			}

			PluginLogger* logger_;
			UnityGfxRenderer unityGfxRendererType_;
		};
	} // namespace runtime
} // namespace b3d::unity_cuda_interop
