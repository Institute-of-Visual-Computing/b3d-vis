
#include "Action.h"
#include "PluginLogger.h"
#include "RendererBase.h"
#include "SyncPrimitiveSampleRenderer.h"
#include "Texture.h"
#include "RenderAPI.h"
#include "RenderingContext.h"

#include "create_action.h"
#include "oneapi/tbb/profiling.h"

#include "NullDebugDrawList.h"


using namespace b3d::renderer;
using namespace b3d::unity_cuda_interop;

enum class CustomActionRenderEventTypes : int
{
	initialize = 0,
	beforeForwardAlpha = 1,

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

	uint64_t currFenceValue = 0;
};

ActionSyncPrimitiveSample::ActionSyncPrimitiveSample()
{
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

	renderer_->initialize(initializationInfo_, DebugInitializationInfo{ std::make_shared<NullDebugDrawList>() });
	isInitialized_ = true;
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
	if (isInitialized_)
	{
		if (eventId == static_cast<int>(CustomActionRenderEventTypes::beforeForwardAlpha))
		{
			logger_->log("Render");
			View v;
			v.colorRt = { .target = testTexture_->getCudaGraphicsResource(),
						  .extent = { static_cast<uint32_t>(testTexture_->getWidth()),
									  static_cast<uint32_t>(testTexture_->getHeight()), 1 } };

			currFenceValue += 1;
			v.fenceValue = currFenceValue;

			renderingContext_->signal(signalPrimitive_.get(), currFenceValue);

			renderer_->render(v);

			renderingContext_->wait(waitPrimitive_.get(), currFenceValue);
		}
	}
	else
	{
		if (eventId == static_cast<int>(CustomActionRenderEventTypes::initialize))
		{
			initialize(data);
		}
	}
}

EXTERN_CREATE_ACTION(ActionSyncPrimitiveSample)
