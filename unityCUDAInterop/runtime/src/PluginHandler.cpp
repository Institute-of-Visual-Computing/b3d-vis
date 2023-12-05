#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

#include "IUnityInterface.h"
#include "IUnityLog.h"

#include "Action.h"
#include "PluginLogger.h"

#include "PluginHandler.h"
#include "PluginRenderEventTypes.h"
#include "RenderAPI.h"

using namespace b3d::unity_cuda_interop;
using namespace b3d::unity_cuda_interop::runtime;

auto PluginHandler::onGraphicsDeviceEvent(const UnityGfxDeviceEventType eventType) -> void
{
    switch (eventType)
    {
    case kUnityGfxDeviceEventInitialize:
        logger_->log("kUnityGfxDeviceEventInitialize");

        if(renderAPI_ == nullptr)
        {
            logger_->log("Try to initialize RenderAPI.");
            renderAPI_ = RenderAPI::createRenderAPI(unityGraphics_->GetRenderer(), unityInterfaces_, logger_.get());
            renderEventIDOffset_ = unityGraphics_->ReserveEventIDRange(renderEventIDCount_);
			logger_->log((("renderEventIDOffset for action is: ") + std::to_string(renderEventIDOffset_)).c_str());
            if(renderAPI_ == nullptr)
            {
                logger_->log("Could not initialize.");
            }
			else
			{
				logger_->log(("RenderAPI created with renderIDOffset: " + std::to_string(renderEventIDOffset_)).c_str());
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
auto PluginHandler::onRenderEventAndData(const int eventID, void* data) -> void
{
    if(eventID < renderEventIDOffset_)
    {
        return;
    }
    assert(eventID >= renderEventIDOffset_);
    if (eventID < renderEventIDOffset_ + renderEventIDCount_)
    {
		// ReSharper disable once CppTooWideScope
		const auto eventType = static_cast<PluginRenderEventTypes>(eventID - renderEventIDOffset_);
        switch (eventType) // NOLINT(clang-diagnostic-switch-enum)
        {
        case PluginRenderEventTypes::rteInitialize:
            renderAPI_->initialize();
            break;
        case PluginRenderEventTypes::rteTeardown:
            tearDown();
            break;
        case PluginRenderEventTypes::actionRegister:
            registerAction(static_cast<Action*>(data));
            break;
        case PluginRenderEventTypes::actionUnregister:
            unregisterAction(static_cast<Action*>(data));
            break;
        default:
            logger_->log(("EventID " + std::to_string(eventID) + " not available.").c_str());
        }
    }
    else
    {
	    for (const auto &registeredAction : registeredActions_)
	    {
		    if(registeredAction->isValidEventID(eventID))
		    {
                registeredAction->renderEventAndData(eventID, data);
                return;
		    }
	    }
    }
    
}

auto PluginHandler::registerGraphicsDeviceEvent(
	const IUnityGraphicsDeviceEventCallback graphicsDeviceEventCallback) const
	-> void
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
	const IUnityGraphicsDeviceEventCallback graphicsDeviceEventCallback) const
	-> void
{
    if (unityGraphics_ != nullptr)
    {
        unityGraphics_->UnregisterDeviceEventCallback(graphicsDeviceEventCallback);
    }
}

// Happens on gfx thread
auto PluginHandler::registerAction(Action* action) -> void
{
    // Check if RTE is running or not
    if(std::ranges::find(registeredActions_, action) != registeredActions_.end())
    {
        // Already in list
        logger_->log("The action was already registered!");
    }

    const auto renderEventIDOffset = unityGraphics_->ReserveEventIDRange(action->getRenderEventIDCount());
	logger_->log((("renderEventIDOffset for action is: ") + std::to_string(renderEventIDOffset)).c_str());
    action->runtimeInitialize(renderEventIDOffset,logger_.get(), renderAPI_.get());
    registeredActions_.push_back(action);
}

// Happens on gfx thread
auto PluginHandler::unregisterAction(Action* action) -> void
{
	// ReSharper disable once CppTooWideScopeInitStatement
	auto [removeBegin, removeEnd] = std::ranges::remove(registeredActions_, action);
    if (removeBegin != registeredActions_.end() && *removeBegin != nullptr)
    {
        (*removeBegin)->runtimeTearDown();
        registeredActions_.erase(removeBegin, removeEnd);
    }
}

// Happens on gfx thread
auto PluginHandler::tearDown() -> void
{
    for (const auto nativeAction : registeredActions_)
    {
        nativeAction->runtimeTearDown();
    }
    registeredActions_.clear();
}

PluginHandler::PluginHandler() : renderEventIDCount_(static_cast<int>(PluginRenderEventTypes::actionRenderEventTypesMax))
{
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
