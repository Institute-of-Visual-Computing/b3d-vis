#include <iostream>

#include "PluginHandler.h"

using namespace b3d::unity_cuda_interop;
using namespace b3d::unity_cuda_interop::runtime;

static std::unique_ptr<PluginHandler> sPluginHandler = std::make_unique<PluginHandler>();

// This callback will be called when graphics device is created, destroyed, reset, etc.
// It is possible to miss the kUnityGfxDeviceEventInitialize event in case plugin is loaded at a later time, when the graphics device is already created.
static auto onGraphicsDeviceEvent(const UnityGfxDeviceEventType eventType) -> void
{
    sPluginHandler->onGraphicsDeviceEvent(eventType);
}

static auto onRenderEventAndData(const int eventID, void* data) -> void
{
    sPluginHandler->onRenderEventAndData(eventID, data);
}

extern "C"
{
    // If exported by a plugin, this function will be called when the plugin is loaded.
	// ReSharper disable once CppInconsistentNaming
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces) -> void
	{
        sPluginHandler->unityPluginLoad(unityInterfaces);
        sPluginHandler->registerGraphicsDeviceEvent(onGraphicsDeviceEvent);
    }

    // If exported by a plugin, this function will be called when the plugin is about to be unloaded.
    // Unity unloads the plugin only, when th Editor gets closed!
    // This function des not get called every time for some reason.
	// ReSharper disable once CppInconsistentNaming
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API UnityPluginUnload() -> void
	{
		sPluginHandler->unityPluginUnload();
	}

	// ReSharper disable once CppInconsistentNaming
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API GetRenderEventAndDataFunc() -> UnityRenderingEventAndData
	{
		return onRenderEventAndData;
	}

	// ReSharper disable once CppInconsistentNaming
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API InitPlugin() -> int
	{
		return 0;
	}

	// ReSharper disable once CppInconsistentNaming
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API GetRenderEventIDOffset() -> int
	{
        return sPluginHandler->getRenderEventIDOffset();
    }
}
