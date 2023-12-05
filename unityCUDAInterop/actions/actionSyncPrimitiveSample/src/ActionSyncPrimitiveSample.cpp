
#include "Action.h"
#include "PluginLogger.h"
#include "RendererBase.h"
#include "SyncPrimitiveSampleRenderer.h"

#include "RenderingContext.h"

using namespace b3d::renderer;
using namespace b3d::unity_cuda_interop;
using namespace b3d::unity_cuda_interop::runtime;

enum class CustomActionRenderEventTypes : int
{
	actionInitialize = 0,
	actionTeardown,
	beforeForwardAlpha,

	customActionRenderEventTypeCount
};

class ActionSyncPrimitiveSample final : public Action
{
public:
	ActionSyncPrimitiveSample();

protected:
	auto initialize(void* data) -> void override;
	auto teardown() -> void override;
	auto customRenderEvent(int eventId, void* data) -> void override;

	std::unique_ptr<SyncPrimitive> waitPrimitive_;
	std::unique_ptr<SyncPrimitive> signalPrimitive_;
	std::unique_ptr<RenderingContext> renderingContext_;
	std::unique_ptr<Texture> testTexture_;
	RendererInitializationInfo initializationInfo_ {};

	// explicite. can be generic
	std::unique_ptr<SyncPrimitiveSampleRenderer> renderer_;
};

ActionSyncPrimitiveSample::ActionSyncPrimitiveSample()
{
	renderEventIDCount_ = static_cast<int>(CustomActionRenderEventTypes::customActionRenderEventTypeCount);
	renderer_ = std::make_unique<SyncPrimitiveSampleRenderer>();
}

auto ActionSyncPrimitiveSample::initialize(void* data) -> void
{
	testTexture_ = renderAPI_->createTexture(data);
	testTexture_->registerCUDA();



	// Get Sync Primitives
	waitPrimitive_ = renderAPI_->createSynchronizationPrimitive();
	signalPrimitive_ = renderAPI_->createSynchronizationPrimitive();
	waitPrimitive_->importToCUDA();
	signalPrimitive_->importToCUDA();

	renderingContext_ = renderAPI_->createRenderingContext();

	initializationInfo_.waitSemaphore = waitPrimitive_->getCudaSemaphore();
	initializationInfo_.signalSemaphore = signalPrimitive_->getCudaSemaphore();
	initializationInfo_.deviceUuid = renderAPI_->getCudaUUID();

	renderer_->initialize(initializationInfo_);
	// Get Cuda Device UUID
	// Create Renderer
}

auto ActionSyncPrimitiveSample::teardown() -> void
{
	renderer_->deinitialize();
	renderer_.reset();
	
	testTexture_.reset();
	waitPrimitive_.reset();
	signalPrimitive_.reset();
}

auto ActionSyncPrimitiveSample::customRenderEvent(int eventId, void* data) -> void
{
	if (eventId == static_cast<int>(CustomActionRenderEventTypes::beforeForwardAlpha))
	{
		renderingContext_->signal(signalPrimitive_.get(), 1);

		logger_->log("Render");
		View v;
		v.colorRt = {
			.target =testTexture_->getCudaGraphicsResource(),
			.extent = { static_cast<uint32_t>(testTexture_->getWidth()), static_cast<uint32_t>(testTexture_->getHeight()), 1 }
			 };
		
		
		renderer_->render(v);

		renderingContext_->wait(waitPrimitive_.get(), 1);

		renderingContext_->signal(signalPrimitive_.get(), 0);
		renderingContext_->signal(waitPrimitive_.get(), 0);
	}
}

extern "C"
{
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API createAction() -> b3d::unity_cuda_interop::Action*
	{
		return new ActionSyncPrimitiveSample();
	}

	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API destroyAction(b3d::unity_cuda_interop::Action* nativeAction) -> void
	{
		delete nativeAction;
	}

	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API
	getRenderEventIDOffset(const b3d::unity_cuda_interop::Action* action) -> int
	{
		if (action == nullptr)
		{
			return -1;
		}
		return action->getRenderEventIDOffset();
	}
}
