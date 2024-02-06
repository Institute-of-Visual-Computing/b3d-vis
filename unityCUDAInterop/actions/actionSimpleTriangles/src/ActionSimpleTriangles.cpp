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

namespace
{
	struct NativeInitData
	{
		NativeTextureData textureData{};
	};

	struct SimpleTriangleNativeRenderingData
	{
		VolumeTransform volumeTransform;
	};
}

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
	auto setTextures(const NativeTextureData* nativeTextureData) -> void;

	std::unique_ptr<SyncPrimitive> waitPrimitive_;
	std::unique_ptr<SyncPrimitive> signalPrimitive_;
	std::unique_ptr<RenderingContext> renderingContext_;

	std::unique_ptr<Texture> colorTexture_;

	RendererInitializationInfo initializationInfo_{};

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

	initializationInfo_.waitSemaphore = waitPrimitive_->getCudaSemaphore();
	initializationInfo_.signalSemaphore = signalPrimitive_->getCudaSemaphore();
	initializationInfo_.deviceUuid = renderAPI_->getCudaUUID();

	renderer_->initialize(
		initializationInfo_,
		DebugInitializationInfo{ std::make_shared<NullDebugDrawList>(), std::make_shared<NullGizmoHelper>() });
	isInitialized_ = true;
}

auto ActionSimpleTriangles::customRenderEvent(int eventId, void* data) -> void
{
	if (isInitialized_ && isReady_ && eventId == static_cast<int>(CustomActionRenderEventTypes::renderEvent))
	{
		if (data == nullptr)
		{
			return;
		}

		const auto nrd = static_cast<NativeRenderingDataWrapper*>(data);

		logger_->log("Render");
		View v;
		v.colorRt = { .target = colorTexture_->getCudaGraphicsResource(),
					  .extent = { static_cast<uint32_t>(colorTexture_->getWidth()),
								  static_cast<uint32_t>(colorTexture_->getHeight()),
								  static_cast<uint32_t>(colorTexture_->getDepth()) } };

		v.cameras[0] = nrd->nativeRenderingData.nativeCameradata[0];
		v.cameras[1] = nrd->nativeRenderingData.nativeCameradata[1];

		v.cameras[0].at.z *= -1.0f;
		v.cameras[0].origin.z *= -1.0f;
		v.cameras[0].up.z *= -1.0f;

		v.cameras[0].dir00.z *= -1.0f;
		v.cameras[0].dirDu.z *= -1.0f;
		v.cameras[0].dirDv.z *= -1.0f;

		v.cameras[1].at.z *= -1.0f;
		v.cameras[1].origin.z *= -1.0f;
		v.cameras[1].up.z *= -1.0f;

		v.cameras[1].dir00.z *= -1.0f;
		v.cameras[1].dirDu.z *= -1.0f;
		v.cameras[1].dirDv.z *= -1.0f;


		v.mode = static_cast<RenderMode>(nrd->nativeRenderingData.eyeCount - 1);

		auto &stnrs = *static_cast<SimpleTriangleNativeRenderingData*>(nrd->additionalDataPointer);

		std::unique_ptr<RendererState> strs_ = std::make_unique<SimpleTriangleRendererState>();
		strs_->worldMatTRS.p = stnrs.volumeTransform.position * owl::vec3f{ 1, 1, -1 };
		/*
		strs_->worldMatTRS.l = owl::LinearSpace3f{
			 owl::Quaternion3f{  stnrs.volumeTransform.rotation.k,
								 owl::vec3f{ -stnrs.volumeTransform.rotation.r, -stnrs.volumeTransform.rotation.i, stnrs.volumeTransform.rotation.j }
			 }
		};
		
		strs_->worldMatTRS.l *= owl::LinearSpace3f::scale(stnrs.volumeTransform.scale);
		*/

		renderer_->setRenderState(std::move(strs_));
		
		currFenceValue += 1;
		v.fenceValue = currFenceValue;

		renderingContext_->signal(signalPrimitive_.get(), currFenceValue);
		renderer_->render(v);

		renderingContext_->wait(waitPrimitive_.get(), currFenceValue);
	}
	else if (eventId == static_cast<int>(CustomActionRenderEventTypes::initializeEvent))
	{
		initialize(data);
	}
	else if (eventId == static_cast<int>(CustomActionRenderEventTypes::setTexturesEvent))
	{
		setTextures(static_cast<NativeTextureData*>(data));
	}
}

auto ActionSimpleTriangles::setTextures(const NativeTextureData* nativeTextureData) -> void
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

	colorTexture_.reset();
	waitPrimitive_.reset();
	signalPrimitive_.reset();
}

EXTERN_CREATE_ACTION(ActionSimpleTriangles)
