#include "IUnityInterface.h"

#include "Action.h"

#include "PluginLogger.h"
#include "Texture.h"
#include "RenderAPI.h"

#include "create_action.h"

using namespace b3d::unity_cuda_interop;

class ActionConcrete : public b3d::unity_cuda_interop::Action
{
public:
	ActionConcrete();

	~ActionConcrete() override;
	auto initialize(void* data) -> void override;
	auto teardown() -> void override;

protected:
	auto customRenderEvent(int eventId, void* data) -> void override;

	
	std::unique_ptr<Texture> testTexture{ nullptr };

};

ActionConcrete::ActionConcrete() : Action()
{
}

ActionConcrete::~ActionConcrete()
{
}
auto ActionConcrete::initialize(void* data) -> void
{
	testTexture = renderAPI_->createTexture(data);
}

auto ActionConcrete::teardown() -> void
{
	logger_->log("Action teardown");
	if (testTexture)
	{
		testTexture.reset();
	}
}

auto ActionConcrete::customRenderEvent(int eventId, void* data) -> void
{
	logger_->log("ConcreteRenderEvent");
}



EXTERN_CREATE_ACTION(ActionConcrete)
