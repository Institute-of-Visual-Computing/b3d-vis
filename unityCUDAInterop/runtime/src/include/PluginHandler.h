#pragma once
#include <memory>
#include <vector>

#include "IUnityGraphics.h"
#include "IUnityInterface.h"

#include "PluginLogger.h"
#include "RenderAPI.h"

struct IUnityLog;

namespace b3d::unity_cuda_interop
{
	class Action;


	namespace runtime
	{
		class PluginHandler
		{
		public:
			PluginHandler();
			auto unityPluginLoad(IUnityInterfaces* unityInterfaces) -> void;
			auto unityPluginUnload() -> void;

			auto onGraphicsDeviceEvent(UnityGfxDeviceEventType eventType) -> void;
			auto onRenderEventAndData(int eventID, void* data) -> void;

			auto registerGraphicsDeviceEvent(IUnityGraphicsDeviceEventCallback graphicsDeviceEventCallback) const
				-> void;
			auto unregisterGraphicsDeviceEvent(IUnityGraphicsDeviceEventCallback graphicsDeviceEventCallback) const
				-> void;


			auto getRenderEventIDOffset() const -> int
			{
				return renderEventIDOffset_;
			}

		private:
			std::unique_ptr<PluginLogger> logger_{ nullptr };
			std::unique_ptr<RenderAPI> renderAPI_{ nullptr };

			IUnityInterfaces* unityInterfaces_{ nullptr };
			IUnityGraphics* unityGraphics_{ nullptr };

			int renderEventIDOffset_{ 0 };
			const int renderEventIDCount_;


			auto registerAction(Action* action) -> void;
			auto unregisterAction(Action* action) -> void;
			auto tearDown() -> void;

			std::vector<Action*> registeredActions_{};
		};
	} // namespace runtime
} // namespace b3d::unity_cuda_interop
