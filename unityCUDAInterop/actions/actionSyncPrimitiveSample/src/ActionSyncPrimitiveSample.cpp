
#include "Action.h"
#include "PluginLogger.h"
#include "RendererBase.h"
#include "SyncPrimitiveSampleRenderer.h"
#include "Texture.h"
#include "RenderAPI.h"
#include "RenderingContext.h"
#include "SharedStructs.h"

#include "create_action.h"
#include "oneapi/tbb/profiling.h"

#include "NullDebugDrawList.h"
#include "NullGizmoHelper.h"


using namespace b3d::renderer;
using namespace b3d::unity_cuda_interop;

namespace
{
	struct NativeInitData
	{
		NativeTextureData textureData{};
	};

} // namespace


enum class CustomActionRenderEventTypes : int
{
	initialize = 0,
	setTexturesEvent,
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
	auto setTextures(const NativeTextureData* nativeTextureData) -> void;

	std::unique_ptr<SyncPrimitive> waitPrimitive_;
	std::unique_ptr<SyncPrimitive> signalPrimitive_;
	std::unique_ptr<RenderingContext> renderingContext_;
	std::unique_ptr<Texture> testTexture_;
	

	// explicite. can be generic
	std::unique_ptr<SyncPrimitiveSampleRenderer> renderer_;
	bool isReady_{ false };

	uint64_t currFenceValue = 0;
};

ActionSyncPrimitiveSample::ActionSyncPrimitiveSample()
{
	renderer_ = std::make_unique<SyncPrimitiveSampleRenderer>();
}

auto ActionSyncPrimitiveSample::initialize(void* data) -> void
{
	const auto initData = static_cast<NativeInitData*>(data);
	if (initData != nullptr)
	{
		setTextures(&initData->textureData);
	}
	testTexture_ = renderAPI_->createTexture(initData->textureData.colorTexture.pointer);
	testTexture_->registerCUDA();

	// Get Sync Primitives
	waitPrimitive_ = renderAPI_->createSynchronizationPrimitive();
	signalPrimitive_ = renderAPI_->createSynchronizationPrimitive();
	waitPrimitive_->importToCUDA();
	signalPrimitive_->importToCUDA();

	renderingContext_ = renderAPI_->createRenderingContext();

	renderingDataWrapper_.data.synchronization.waitSemaphore = waitPrimitive_->getCudaSemaphore();
	renderingDataWrapper_.data.synchronization.signalSemaphore = signalPrimitive_->getCudaSemaphore();
	renderingDataWrapper_.data.rendererInitializationInfo.deviceUuid = renderAPI_->getCudaUUID();

	renderer_->initialize(
		&renderingDataWrapper_.buffer,
		DebugInitializationInfo{ std::make_shared<NullDebugDrawList>(), std::make_shared<NullGizmoHelper>() });
	isInitialized_ = true;
}

auto ActionSyncPrimitiveSample::teardown() -> void
{
	isReady_ = false;
	isInitialized_ = false;
	renderer_->deinitialize();
	cudaDeviceSynchronize();
	renderer_.reset();
	
	testTexture_.reset();
	waitPrimitive_.reset();
	signalPrimitive_.reset();
}

auto ActionSyncPrimitiveSample::customRenderEvent(int eventId, void* data) -> void
{
	if (isInitialized_ && isReady_ && eventId == static_cast<int>(CustomActionRenderEventTypes::beforeForwardAlpha))
	{
		logger_->log("Render");

		renderingDataWrapper_.data.renderTargets.colorRt = {
			.target = testTexture_->getCudaGraphicsResource(),
					  .extent = { static_cast<uint32_t>(testTexture_->getWidth()),
								  static_cast<uint32_t>(testTexture_->getHeight()), 1 } };

		currFenceValue += 1;
		renderingDataWrapper_.data.synchronization.fenceValue = currFenceValue;

		renderingContext_->signal(signalPrimitive_.get(), currFenceValue);

		renderer_->render();

		renderingContext_->wait(waitPrimitive_.get(), currFenceValue);
		
	}
	else if (eventId == static_cast<int>(CustomActionRenderEventTypes::initialize))
	{
		initialize(data);
	}
	else if (eventId == static_cast<int>(CustomActionRenderEventTypes::setTexturesEvent))
	{
		setTextures(static_cast<NativeTextureData*>(data));
	}
}

auto ActionSyncPrimitiveSample::setTextures(const NativeTextureData* nativeTextureData) -> void
{
	isReady_ = false;
	const auto ntd = *nativeTextureData;
	if (ntd.colorTexture.extent.depth > 0)
	{
		auto newColorTexture = renderAPI_->createTexture(ntd.colorTexture.pointer);
		newColorTexture->registerCUDA();
		cudaDeviceSynchronize();
		testTexture_.swap(newColorTexture);
		cudaDeviceSynchronize();
	}

	if (ntd.depthTexture.extent.depth > 0)
	{
	}
	isReady_ = true;
}

EXTERN_CREATE_ACTION(ActionSyncPrimitiveSample)
