#pragma once
#include <map>
#include <memory>
#include <vector>

#include "IUnityGraphics.h"

struct IUnityLog;

namespace b3d::unity_cuda_interop
{
	class PluginLogger;
	class RenderAPI;

	class Action;

	class PluginHandler
	{
	public:
		PluginHandler();
		auto unityPluginLoad(IUnityInterfaces* unityInterfaces) -> void;
		auto unityPluginUnload() -> void;

		auto onGraphicsDeviceEvent(UnityGfxDeviceEventType eventType) -> void;
		auto onRenderEventAndData(int eventID, void* data) const -> void;

		auto registerGraphicsDeviceEvent(IUnityGraphicsDeviceEventCallback graphicsDeviceEventCallback) const
			-> void;
		auto unregisterGraphicsDeviceEvent(IUnityGraphicsDeviceEventCallback graphicsDeviceEventCallback) const
			-> void;

		auto registerAction(Action* action) -> void;
		auto unregisterAction(Action* action) -> void;


	private:
		std::unique_ptr<PluginLogger> logger_{ nullptr };
		std::unique_ptr<RenderAPI> renderAPI_{ nullptr };

		std::unique_ptr<Action> registeredAction_{ nullptr };

		std::map<int, Action*> registeredActions_{};
		
		IUnityInterfaces* unityInterfaces_{ nullptr };
		IUnityGraphics* unityGraphics_{ nullptr };
	};
} // namespace b3d::unity_cuda_interop
