
#include "Action.h"

#include "PluginHandler.h"
#include "PluginLogger.h"
#include "RenderAPI.h"

#include <iostream>

using namespace b3d::unity_cuda_interop;

PluginHandler::PluginHandler()
{
}

auto PluginHandler::onGraphicsDeviceEvent(const UnityGfxDeviceEventType eventType) -> void
{
	switch (eventType)
	{
	case kUnityGfxDeviceEventInitialize:
		logger_->log("kUnityGfxDeviceEventInitialize");

		if (renderAPI_ == nullptr)
		{
			logger_->log("Try to initialize RenderAPI.");
			renderAPI_ = RenderAPI::createRenderAPI(unityGraphics_->GetRenderer(), unityInterfaces_, logger_.get());
			
			//logger_->log((("renderEventIDOffset for action is: ") + std::to_string(renderEventIDOffset_)).c_str());
			if (renderAPI_ == nullptr)
			{
				logger_->log("Could not initialize.");
			}
			else
			{
				renderAPI_->initialize();
				// logger_->log(("RenderAPI created with renderIDOffset: " + std::to_string(renderEventIDOffset_)).c_str());
			}
		}
		else
		{
			logger_->log("Backend is not D3D11!");
		}
		break;
	case kUnityGfxDeviceEventShutdown:
		// Gets called multiple times sometimes
		logger_->log("kUnityGfxDeviceEventShutdown");
		break;
	case kUnityGfxDeviceEventBeforeReset:
		logger_->log("kUnityGfxDeviceEventBeforeReset");
		break;
	case kUnityGfxDeviceEventAfterReset:
		logger_->log("kUnityGfxDeviceEventAfterReset");
		break;
	}
}

// Happens on gfx thread
auto PluginHandler::onRenderEventAndData(const int eventID, void* data) const -> void
{
	for (const auto &registeredAction : registeredActions_)
	{
		if (registeredAction.second->containsEventId(eventID))
		{
			registeredAction.second->renderEventAndData(eventID, data);
			return;
		}
	}
}

auto PluginHandler::registerGraphicsDeviceEvent(
	const IUnityGraphicsDeviceEventCallback graphicsDeviceEventCallback) const -> void
{
	if (unityGraphics_ != nullptr)
	{
		logger_->log("Register Device Event callback for graphics device");

		unityGraphics_->RegisterDeviceEventCallback(graphicsDeviceEventCallback);

		// In case the graphics device was already loaded the Init event is not raised. Do it manually here.
		graphicsDeviceEventCallback(kUnityGfxDeviceEventInitialize);
	}
}

auto PluginHandler::unregisterGraphicsDeviceEvent(
	const IUnityGraphicsDeviceEventCallback graphicsDeviceEventCallback) const -> void
{
	if (unityGraphics_ != nullptr)
	{
		unityGraphics_->UnregisterDeviceEventCallback(graphicsDeviceEventCallback);
	}
}

auto PluginHandler::unregisterAction(Action* action) -> void
{

	if (action == nullptr || registeredActions_.erase(action->getRenderEventIDOffset()) < 1)
	{
		logger_->log("Action was not registered.");
		return;
	}

	action->unregisterAction();
}


auto PluginHandler::registerAction(Action* action) -> void
{
	if (action == nullptr)
	{
		logger_->log("Action is nullptr");
		return;
	}

	if (action->getRenderEventIDOffset() > -1)
	{
		logger_->log("Action is already registered.");
		if (registeredActions_.contains(action->getRenderEventIDOffset()))
		{
			logger_->log("Action is already in registered collection.");
			return;
		}
	}
	else
	{
		logger_->log("Register action");
		action->registerAction(unityGraphics_->ReserveEventIDRange(Action::eventIdCount), logger_.get(),
							   renderAPI_.get());
	}

	registeredActions_[action->getRenderEventIDOffset()] = action;
}


auto PluginHandler::unityPluginLoad(IUnityInterfaces* unityInterfaces) -> void
{
	if (unityInterfaces_ != nullptr)
	{
		if (logger_ != nullptr)
		{
			logger_->log("Unity tries to load the plugin a seconds time. Which should not happen!\n");
		}
		else
		{
			std::cerr << "Unity tries to load the plugin a second time. Which should not happen!\n";
		}
		return;
	}
	unityInterfaces_ = unityInterfaces;

	logger_ = std::make_unique<PluginLogger>(unityInterfaces_->Get<IUnityLog>());
	if (logger_ == nullptr)
	{
		std::cerr << "Could not retrieve IUnityLog from unityInterfaces.\n";
	}
	else
	{
		logger_->log("UnityNativePluginSequence loaded.");
	}

	unityGraphics_ = unityInterfaces_->Get<IUnityGraphics>();
}

auto PluginHandler::unityPluginUnload() -> void
{
	if (unityInterfaces_ == nullptr)
	{
		std::cout << "Unity tries to unload the plugin UnityNativePluginSequence. But it was not loaded.\n";
		return;
	}

	logger_->log("Unload plugin UnityNativePluginSequence.");

	unityGraphics_ = nullptr;
	unityInterfaces_ = nullptr;
}
