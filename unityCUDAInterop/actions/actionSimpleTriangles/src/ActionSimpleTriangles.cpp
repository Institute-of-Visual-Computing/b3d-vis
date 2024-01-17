#include "Action.h"

#include "NullDebugDrawList.h"
#include "NullGizmoHelper.h"
#include "PluginLogger.h"
#include "RenderAPI.h"
#include "RendererBase.h"
#include "RenderingContext.h"
#include "SimpleTrianglesRenderer.h"
#include "Texture.h"

#include "create_action.h"


using namespace b3d::renderer;
using namespace b3d::unity_cuda_interop;

namespace
{
	struct NativeRenderingData
	{
		int eyeCount{ 1 };
		Camera nativeCameradata[2];
		NativeCube nativeCube{};
	};

	struct UnityInputTexture
	{
		void* pointer{ nullptr };
		Extent exent{ 0, 0, 0 };
	};

	struct NativeTextureData
	{
		UnityInputTexture colorTexture{};
		UnityInputTexture depthTexture{};
	};

	struct NativeInitData
	{
		NativeTextureData textureData{};
	};
} // namespace

enum class CustomActionRenderEventTypes : int
{
	initialize = 0,
	setTextures,
	beforeForwardAlpha,

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
	if (isInitialized_ && isReady_ && eventId == static_cast<int>(CustomActionRenderEventTypes::beforeForwardAlpha))
	{
		if (data == nullptr)
		{
			return;
		}

		const auto nrd = static_cast<NativeRenderingData*>(data);

		logger_->log("Render");
		View v;
		v.colorRt = { .target = colorTexture_->getCudaGraphicsResource(),
					  .extent = { static_cast<uint32_t>(colorTexture_->getWidth()),
								  static_cast<uint32_t>(colorTexture_->getHeight()),
								  static_cast<uint32_t>(colorTexture_->getDepth()) } };

		v.cameras[0] = nrd->nativeCameradata[0];
		v.cameras[1] = nrd->nativeCameradata[0];

		v.cameras[0].at.z *= -1.0f;
		v.cameras[0].origin.z *= -1.0f;
		v.cameras[0].up.z *= -1.0f;

		v.cameras[1].at.z *= -1.0f;
		v.cameras[1].origin.z *= -1.0f;
		v.cameras[1].up.z *= -1.0f;

		v.mode = static_cast<RenderMode>(nrd->eyeCount - 1);

		nrd->nativeCube.position.z *= -1.0f;

		nrd->nativeCube.rotation =
			owl::Quaternion3f{ nrd->nativeCube.rotation.k,
							   owl::vec3f{ -nrd->nativeCube.rotation.r, -nrd->nativeCube.rotation.i,
										   nrd->nativeCube.rotation.j } };

		renderer_->setCubeVolumeTransform(&nrd->nativeCube);

		currFenceValue += 1;
		v.fenceValue = currFenceValue;

		renderingContext_->signal(signalPrimitive_.get(), currFenceValue);
		renderer_->render(v);

		renderingContext_->wait(waitPrimitive_.get(), currFenceValue);
	}
	else if (eventId == static_cast<int>(CustomActionRenderEventTypes::initialize))
	{
		initialize(data);
	}
	else if (eventId == static_cast<int>(CustomActionRenderEventTypes::setTextures))
	{
		setTextures(static_cast<NativeTextureData*>(data));
	}
}

auto ActionSimpleTriangles::setTextures(const NativeTextureData* nativeTextureData) -> void
{
	isReady_ = false;
	const auto ntd = *nativeTextureData;
	if (ntd.colorTexture.exent.depth > 0)
	{
		auto newColorTexture = renderAPI_->createTexture(ntd.colorTexture.pointer);
		newColorTexture->registerCUDA();
		cudaDeviceSynchronize();
		colorTexture_.swap(newColorTexture);
		cudaDeviceSynchronize();
	}

	if (ntd.depthTexture.exent.depth > 0)
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
