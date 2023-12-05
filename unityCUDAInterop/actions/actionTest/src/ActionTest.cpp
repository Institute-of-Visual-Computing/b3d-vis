#include <vector>

#include <format>
#include "Action.h"
#include "PluginLogger.h"

#include "d3d11_4.h"

#include "../../../runtime/src/include/RenderAPI/RenderAPI_D3D11.h"
#include "../../../runtime/src/include/SyncPrimitive/SyncPrimitive_D3D11.h"
#include "IUnityGraphicsD3D11.h"

using namespace b3d::unity_cuda_interop;
using namespace b3d::unity_cuda_interop::runtime;

enum class CustomActionRenderEventTypes :int 
{
	actionInitialize = 0,
	actionTeardown,
	actionUserUpdate,
	customActionRenderEventTypeCount
};

class ActionTest final : public Action
{
public:
	ActionTest();
protected:
	auto initialize(void* data) -> void override;
	auto teardown() -> void override;
	auto customRenderEvent(int eventId, void* data) -> void override;

	std::unique_ptr<Texture> testTexture{ nullptr };
};

ActionTest::ActionTest()
{
	renderEventIDCount_ = static_cast<int>(CustomActionRenderEventTypes::customActionRenderEventTypeCount);
}

auto ActionTest::initialize(void* data)-> void
{
	testTexture = renderAPI_->createTexture(data);
	logger_->log("Action initialize");
	
	logger_->log(std::format("Texture is valid {:6}", testTexture->isValid()).c_str());
	
}

auto ActionTest::teardown() -> void
{
	logger_->log("Action teardown");
	if(testTexture)
	{
		testTexture.reset();
	}
}

auto ActionTest::customRenderEvent(int eventId, void* data) -> void
{
	logger_->log("Action Renderevent");
}

extern "C"
{
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API createAction() -> b3d::unity_cuda_interop::Action*
	{
		return new ActionTest();
	}

	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API destroyAction(b3d::unity_cuda_interop::Action* nativeAction) -> void
	{
		delete nativeAction;
	}
	
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API getRenderEventIDOffset(const b3d::unity_cuda_interop::Action* action) -> int
	{
		if(action == nullptr)
		{
			return -1;
		}
		return action->getRenderEventIDOffset();
	}
}
