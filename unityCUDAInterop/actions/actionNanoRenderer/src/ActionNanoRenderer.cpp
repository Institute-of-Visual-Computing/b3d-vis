#include "Action.h"

#include "NullDebugDrawList.h"
#include "NullGizmoHelper.h"
#include "PluginLogger.h"
#include "RenderAPI.h"
#include "RendererBase.h"
#include "RenderingContext.h"
#include "SharedStructs.h"
#include "Texture.h"

#include "NanoRenderer.h"


// This line is crucial and must stay. Should be one of the last include. But in any case after the include of Action.
#include "create_action.h"

using namespace b3d::renderer;
using namespace b3d::unity_cuda_interop;

enum class NanoRenderEventTypes : int
{
	initializeEvent = 0,
	setTexturesEvent,
	renderEvent,
	customActionRenderEventTypeCount
};

class ActionNanoRenderer final : public Action
{
public:
	ActionNanoRenderer();
	auto initialize(void* data) -> void override;
	auto teardown() -> void override;

protected:
	auto customRenderEvent(int eventId, void* data) -> void override;
	auto setTextures(const RenderingDataBuffer& renderingDataBuffer) -> void;

	std::unique_ptr<SyncPrimitive> waitPrimitive_;
	std::unique_ptr<SyncPrimitive> signalPrimitive_;
	std::unique_ptr<RenderingContext> renderingContext_;

	std::unique_ptr<NanoRenderer> renderer_;

	std::unique_ptr<Texture> colorTexture_;
	std::unique_ptr<Texture> depthTexture_;
	
	std::unique_ptr<Texture> colorMapsTexture_;
	std::unique_ptr<Texture> transferFunctionTexture_;

	bool isReady_{ false };
	uint64_t currFenceValue = 0;
};

ActionNanoRenderer::ActionNanoRenderer()
{
	renderer_ = std::make_unique<NanoRenderer>();

}

auto ActionNanoRenderer::initialize(void* data) -> void
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

	const auto colorMapsTexture = rdb.get<UnityTexture>("colorMapsTexture");
	const auto coloringInfo = rdb.get<UnityColoringInfo>("coloringInfo");

	colorMapsTexture_ = renderAPI_->createTexture(colorMapsTexture->texturePointer);
	if (colorMapsTexture_->isValid())
	{
		colorMapsTexture_->registerCUDA();
		cudaDeviceSynchronize();
		renderingDataWrapper_.data.colorMapTexture = { .target = colorMapsTexture_->getCudaGraphicsResource(),
													   .extent = {
														   static_cast<uint32_t>(colorMapsTexture_->getWidth()),
														   static_cast<uint32_t>(colorMapsTexture_->getHeight()),
														   static_cast<uint32_t>(colorMapsTexture_->getDepth()) } };

		renderingDataWrapper_.data.coloringInfo = *coloringInfo;
	}
	else
	{
		logger_->log("Nano action color texture not valid.");
	}
	
	const auto transferFunctionTexture = rdb.get<UnityTexture>("transferFunctionTexture");
	transferFunctionTexture_ = renderAPI_->createTexture(transferFunctionTexture->texturePointer);
		if (transferFunctionTexture_->isValid())
		{
			
		transferFunctionTexture_->registerCUDA();
		cudaDeviceSynchronize();
		renderingDataWrapper_.data.transferFunctionTexture = {
			.target = transferFunctionTexture_->getCudaGraphicsResource(),
			.extent = { static_cast<uint32_t>(transferFunctionTexture_->getWidth()),
						static_cast<uint32_t>(transferFunctionTexture_->getHeight()),
						static_cast<uint32_t>(transferFunctionTexture_->getDepth()) }
		};
	}
	renderer_->initialize(
		&renderingDataWrapper_.buffer,
		DebugInitializationInfo{ std::make_shared<NullDebugDrawList>(), std::make_shared<NullGizmoHelper>() });
	cudaDeviceSynchronize();
	isInitialized_ = true;
	logger_->log("Nano action initialized");
}

auto ActionNanoRenderer::teardown() -> void
{
	isReady_ = false;
	isInitialized_ = false;
	renderer_->deinitialize();
	cudaDeviceSynchronize();
	renderer_.reset();

	transferFunctionTexture_.reset();
	colorMapsTexture_.reset();
	depthTexture_.reset();
	colorTexture_.reset();
	waitPrimitive_.reset();
	signalPrimitive_.reset();
}

auto ActionNanoRenderer::customRenderEvent(int eventId, void* data) -> void
{
	logger_->log("Nano custom render event");
	if (isInitialized_ && isReady_ && colorTexture_->isValid() &&
		eventId == static_cast<int>(NanoRenderEventTypes::renderEvent))
	{
		logger_->log("Nano render");
		const RenderingDataBuffer rdb{ unityDataSchema, 1, data };
		
		renderingDataWrapper_.data.renderTargets.colorRt = { .target = colorTexture_->getCudaGraphicsResource(),
					  .extent = { static_cast<uint32_t>(colorTexture_->getWidth()),
								  static_cast<uint32_t>(colorTexture_->getHeight()),
								  static_cast<uint32_t>(colorTexture_->getDepth()) } };

		const auto coloringInfo = rdb.get<UnityColoringInfo>("coloringInfo");
		renderingDataWrapper_.data.coloringInfo = *coloringInfo;

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

		renderingDataWrapper_.data.volumeTransform.worldMatTRS.l = owl::LinearSpace3f{ owl::Quaternion3f{
			volumeTransform->rotation.k,
			owl::vec3f{ -volumeTransform->rotation.r, -volumeTransform->rotation.i, volumeTransform->rotation.j } } };

		renderingDataWrapper_.data.volumeTransform.worldMatTRS.l *= owl::LinearSpace3f::scale(volumeTransform->scale);

		currFenceValue += 1;
		renderingDataWrapper_.data.synchronization.fenceValue = currFenceValue;

		renderingContext_->signal(signalPrimitive_.get(), currFenceValue);

		renderer_->render();
		renderingContext_->wait(waitPrimitive_.get(), currFenceValue);
	}
	else if (eventId == static_cast<int>(NanoRenderEventTypes::initializeEvent))
	{
		logger_->log("Nano init event");
		initialize(data);
	}
	else if (eventId == static_cast<int>(NanoRenderEventTypes::setTexturesEvent))
	{
		logger_->log("Nano setTexturesEvent");
		const RenderingDataBuffer rdb{ unityDataSchema, 1, data };
		setTextures(rdb);
	}
}

auto ActionNanoRenderer::setTextures(const RenderingDataBuffer& renderingDataBuffer) -> void
{
	isReady_ = false;
	const auto& unityRenderTargets = renderingDataBuffer.get<UnityRenderTargets>("renderTargets");
	if (unityRenderTargets->colorTexture.textureExtent.depth > 0)
	{
		auto newColorTexture = renderAPI_->createTexture(unityRenderTargets->colorTexture.texturePointer);
		if (newColorTexture->isValid())
		{
			newColorTexture->registerCUDA();
			cudaDeviceSynchronize();
			colorTexture_.swap(newColorTexture);
			cudaDeviceSynchronize();	
		}
	}

	if (unityRenderTargets->depthTexture.textureExtent.depth > 0)
	{
		auto newDepthTexture = renderAPI_->createTexture(unityRenderTargets->depthTexture.texturePointer);
		if (newDepthTexture->isValid())
		{
			newDepthTexture->registerCUDA();
			cudaDeviceSynchronize();
			depthTexture_.swap(newDepthTexture);
			cudaDeviceSynchronize();
		}
	}
	isReady_ = true;
}

// This line is crucial and must stay. Replace type with your newly created action type.
EXTERN_CREATE_ACTION(ActionNanoRenderer)
