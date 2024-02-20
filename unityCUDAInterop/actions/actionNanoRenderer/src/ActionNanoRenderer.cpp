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


namespace
{
	struct NativeInitData
	{
		NativeTextureData textureData{};
	};

	struct NanoRendererNativeRenderingData
	{
		VolumeTransform volumeTransform;
	};
} // namespace

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
	auto setTextures(const NativeTextureData* nativeTextureData) -> void;

	std::unique_ptr<SyncPrimitive> waitPrimitive_;
	std::unique_ptr<SyncPrimitive> signalPrimitive_;
	std::unique_ptr<RenderingContext> renderingContext_;

	std::unique_ptr<NanoRenderer> renderer_;

	std::unique_ptr<Texture> colorTexture_;
	std::unique_ptr<Texture> depthTexture_;


	bool isReady_{ false };
	uint64_t currFenceValue = 0;
};

ActionNanoRenderer::ActionNanoRenderer()
{
	renderer_ = std::make_unique<NanoRenderer>();

}

auto ActionNanoRenderer::initialize(void* data) -> void
{
	const auto initData = static_cast<NativeInitData*>(data);
	if (initData != nullptr)
	{
		setTextures(&initData->textureData);
	}

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

auto ActionNanoRenderer::teardown() -> void
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
	renderingContext_.reset();
}

auto ActionNanoRenderer::customRenderEvent(int eventId, void* data) -> void
{
	if (isInitialized_ && isReady_ && eventId == static_cast<int>(NanoRenderEventTypes::renderEvent))
	{
		if (data == nullptr)
		{
			return;
		}

		const auto nrd = static_cast<NativeRenderingDataWrapper*>(data);
		
		renderingDataWrapper_.data.renderTargets.colorRt = { .target = colorTexture_->getCudaGraphicsResource(),
					  .extent = { static_cast<uint32_t>(colorTexture_->getWidth()),
								  static_cast<uint32_t>(colorTexture_->getHeight()),
								  static_cast<uint32_t>(colorTexture_->getDepth()) } };

		renderingDataWrapper_.data.view.cameras[0] = nrd->nativeRenderingData.nativeCameradata[0];
		renderingDataWrapper_.data.view.cameras[1] = nrd->nativeRenderingData.nativeCameradata[1];

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

		renderingDataWrapper_.data.view.mode = static_cast<RenderMode>(nrd->nativeRenderingData.eyeCount - 1);

		auto& stnrs = *static_cast<NanoRendererNativeRenderingData*>(nrd->additionalDataPointer);

		renderingDataWrapper_.data.volumeTransform.worldMatTRS.p = stnrs.volumeTransform.position * owl::vec3f{ 1, 1, -1 };
		
		renderingDataWrapper_.data.volumeTransform.worldMatTRS.l = owl::LinearSpace3f{
			owl::Quaternion3f{
				stnrs.volumeTransform.rotation.k,
				owl::vec3f{ -stnrs.volumeTransform.rotation.r, -stnrs.volumeTransform.rotation.i,stnrs.volumeTransform.rotation.j }
			}
		};

		renderingDataWrapper_.data.volumeTransform.worldMatTRS.l *=
			owl::LinearSpace3f::scale(stnrs.volumeTransform.scale);

		currFenceValue += 1;

		renderingDataWrapper_.data.synchronization.fenceValue = currFenceValue;

		renderingContext_->signal(signalPrimitive_.get(), currFenceValue);
		renderer_->render();

		renderingContext_->wait(waitPrimitive_.get(), currFenceValue);
	}
	else if (eventId == static_cast<int>(NanoRenderEventTypes::initializeEvent))
	{
		initialize(data);
	}
	else if (eventId == static_cast<int>(NanoRenderEventTypes::setTexturesEvent))
	{
		setTextures(static_cast<NativeTextureData*>(data));
	}
}

auto ActionNanoRenderer::setTextures(const NativeTextureData* nativeTextureData) -> void
{
	isReady_ = false;
	const auto ntd = *nativeTextureData;
	if (ntd.colorTexture.extent.depth > 0)
	{
		auto newColorTexture = renderAPI_->createTexture(ntd.colorTexture.pointer);
		newColorTexture->registerCUDA();
		cudaDeviceSynchronize();
		colorTexture_.swap(newColorTexture);
		cudaDeviceSynchronize();
	}

	if (ntd.depthTexture.extent.depth > 0)
	{
		auto newDepthTexture = renderAPI_->createTexture(ntd.depthTexture.pointer);
		newDepthTexture->registerCUDA();
		cudaDeviceSynchronize();
		depthTexture_.swap(newDepthTexture);
		cudaDeviceSynchronize();
	}
	isReady_ = true;
}

// This line is crucial and must stay. Replace type with your newly created action type.
EXTERN_CREATE_ACTION(ActionNanoRenderer)
