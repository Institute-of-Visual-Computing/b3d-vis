#include "Action.h"

#include "NullDebugDrawList.h"
#include "NullGizmoHelper.h"
#include "PluginLogger.h"
#include "RenderAPI.h"
#include "RendererBase.h"
#include "RenderingContext.h"
#include "SimpleTrianglesRenderer.h"
#include "Texture.h"

#include "SharedStructs.h"

#include "create_action.h"


using namespace b3d::renderer;
using namespace b3d::unity_cuda_interop;

enum class CustomActionRenderEventTypes : int
{
	initializeEvent = 0,
	setTexturesEvent,
	renderEvent,
	customActionRenderEventTypeCount
};

class ActionSimpleTriangles final : public Action
{
public:
	ActionSimpleTriangles();
	auto initialize(void* data) -> void override;
	auto teardown() -> void override;

protected:
	auto customRenderEvent(int eventId, void* data) -> void override;
	auto setTextures(const RenderingDataBuffer& renderingDataBuffer) -> void;

	std::unique_ptr<SyncPrimitive> waitPrimitive_;
	std::unique_ptr<SyncPrimitive> signalPrimitive_;
	std::unique_ptr<RenderingContext> renderingContext_;

	std::unique_ptr<Texture> colorTexture_;
	std::unique_ptr<Texture> depthTexture_;

	// explicite. can be generic
	std::unique_ptr<SimpleTrianglesRenderer> renderer_;
	bool isReady_{ false };
	uint64_t currFenceValue = 0;
};

ActionSimpleTriangles::ActionSimpleTriangles()
{
	renderer_ = std::make_unique<SimpleTrianglesRenderer>();
}

auto ActionSimpleTriangles::initialize(void* data) -> void
{
	if (data == nullptr)
	{
		return;
	}
	
	const RenderingDataBuffer rdb{ unityDataSchema, 1, data };
	setTextures(rdb);

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

auto ActionSimpleTriangles::customRenderEvent(int eventId, void* data) -> void
{

	if (isInitialized_ && isReady_ && eventId == static_cast<int>(CustomActionRenderEventTypes::renderEvent))
	{
		const RenderingDataBuffer rdb{ unityDataSchema, 1, data };

		logger_->log("Render");
		renderingDataWrapper_.data.renderTargets.colorRt = { .target = colorTexture_->getCudaGraphicsResource(),
					  .extent = { static_cast<uint32_t>(colorTexture_->getWidth()),
								  static_cast<uint32_t>(colorTexture_->getHeight()),
								  static_cast<uint32_t>(colorTexture_->getDepth()) } };

		const auto& view = rdb.get<View>("view");
		renderingDataWrapper_.data.view.cameras[0] = view->cameras[0];
		renderingDataWrapper_.data.view.cameras[1] = view->cameras[1];

		renderingDataWrapper_.data.view.cameras[0].at.z *= -1.0f;
		renderingDataWrapper_.data.view.cameras[0].origin.z *= -1.0f;
		renderingDataWrapper_.data.view.cameras[0].up.z *= -1.0f;

		renderingDataWrapper_.data.view.cameras[0].dir00.z *= -1.0f;
		renderingDataWrapper_.data.view.cameras[0].dirDu.z *= -1.0f;
		renderingDataWrapper_.data.view.cameras[0].dirDv.z *= -1.0f;

		renderingDataWrapper_.data.view.cameras[1].at.z *= -1.0f;
		renderingDataWrapper_.data.view.cameras[1].origin.z *= -1.0f;
		renderingDataWrapper_.data.view.cameras[1].up.z *= -1.0f;

		renderingDataWrapper_.data.view.cameras[1].dir00.z *= -1.0f;
		renderingDataWrapper_.data.view.cameras[1].dirDu.z *= -1.0f;
		renderingDataWrapper_.data.view.cameras[1].dirDv.z *= -1.0f;

		renderingDataWrapper_.data.view.mode = view->mode;

		
		const auto& volumeTransform = rdb.get<UnityVolumeTransform>("volumeTransform");
		renderingDataWrapper_.data.volumeTransform.worldMatTRS.p = volumeTransform->position * owl::vec3f{ 1, 1, -1 };
		
		renderingDataWrapper_.data.volumeTransform.worldMatTRS.l = owl::LinearSpace3f{
			 owl::Quaternion3f{  volumeTransform->rotation.k,
			owl::vec3f{ -volumeTransform->rotation.r, -volumeTransform->rotation.i, volumeTransform->rotation.j }
			 }
		};
		
		renderingDataWrapper_.data.volumeTransform.worldMatTRS.l *= owl::LinearSpace3f::scale(volumeTransform->scale);
		
		currFenceValue += 1;
		renderingDataWrapper_.data.synchronization.fenceValue = currFenceValue;

		renderingContext_->signal(signalPrimitive_.get(), currFenceValue);
		renderer_->render();

		renderingContext_->wait(waitPrimitive_.get(), currFenceValue);
	}
	else if (eventId == static_cast<int>(CustomActionRenderEventTypes::initializeEvent))
	{
		initialize(data);
	}
	else if (eventId == static_cast<int>(CustomActionRenderEventTypes::setTexturesEvent))
	{
		const RenderingDataBuffer rdb{ unityDataSchema, 1, data };
		setTextures(rdb);
	}
}

auto ActionSimpleTriangles::setTextures(const RenderingDataBuffer& renderingDataBuffer) -> void
{
	isReady_ = false;
	
	const auto& unityRenderTargets = renderingDataBuffer.get<UnityRenderTargets>("renderTargets");
	
	if (unityRenderTargets->colorTexture.textureExtent.depth > 0)
	{
		auto newColorTexture = renderAPI_->createTexture(unityRenderTargets->colorTexture.texturePointer);
		newColorTexture->registerCUDA();
		cudaDeviceSynchronize();
		colorTexture_.swap(newColorTexture);
		cudaDeviceSynchronize();
	}

	if (unityRenderTargets->depthTexture.textureExtent.depth > 0)
	{
		auto newDepthTexture = renderAPI_->createTexture(unityRenderTargets->depthTexture.texturePointer);
		newDepthTexture->registerCUDA();
		cudaDeviceSynchronize();
		depthTexture_.swap(newDepthTexture);
		cudaDeviceSynchronize();
	}
	isReady_ = true;
}

auto ActionSimpleTriangles::teardown() -> void
{
	isReady_ = false;
	isInitialized_ = false;
	renderer_->deinitialize();
	cudaDeviceSynchronize();
	renderer_.reset();

	depthTexture_.reset();
	colorTexture_.reset();
	waitPrimitive_.reset();
	signalPrimitive_.reset();
}

EXTERN_CREATE_ACTION(ActionSimpleTriangles)
