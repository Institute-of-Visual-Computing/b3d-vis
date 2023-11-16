#pragma once
#include <memory>

#include "Texture.h"

class PluginLogger;

enum UnityGfxRenderer : int;
struct IUnityInterfaces;

class RenderAPI
{
	public:
		static std::unique_ptr<RenderAPI> createRenderAPI(UnityGfxRenderer unityGfxRenderer, IUnityInterfaces* unityInterfaces, PluginLogger *logger);

		RenderAPI() = delete;

		virtual ~RenderAPI();

		// Use this only on the renderThread
		virtual void initialize() = 0;

		virtual std::unique_ptr<Texture> createTexture(void *unityNativeTexturePointer) = 0;

	protected:
		RenderAPI(PluginLogger* logger) : logger_(logger) {}

		PluginLogger* logger_;
		UnityGfxRenderer unityGfxRendererType_;
};
