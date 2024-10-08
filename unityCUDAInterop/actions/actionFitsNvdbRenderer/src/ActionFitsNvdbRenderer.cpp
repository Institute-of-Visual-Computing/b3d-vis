#include "Action.h"

#include "NullDebugDrawList.h"
#include "NullGizmoHelper.h"
#include "PluginLogger.h"
#include "RenderAPI.h"
#include "RendererBase.h"
#include "RenderingContext.h"
#include "SharedStructs.h"
#include "Texture.h"

#include "FitsNvdbRenderer.h"


// This line is crucial and must stay. Should be one of the last include. But in any case after the include of Action.
#include <cuda_d3d11_interop.h>
#include <filesystem>

#include "IUnityGraphicsD3D11.h"
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

class ActionFitsNvdbRenderer final : public Action
{
public:
	ActionFitsNvdbRenderer();
	auto initialize(void* data) -> void override;
	auto teardown() -> void override;

protected:
	auto customRenderEvent(int eventId, void* data) -> void override;
	auto setTextures(const RenderingDataBuffer& renderingDataBuffer) -> void;

	std::unique_ptr<FitsNvdbRenderer> renderer_;

	std::unique_ptr<Texture> colorTexture_;
	std::unique_ptr<Texture> depthTexture_;

	std::unique_ptr<Texture> colorMapsTexture_;
	std::unique_ptr<Texture> transferFunctionTexture_;


	std::atomic_flag resourceCreated = ATOMIC_FLAG_INIT;

	bool isReady_{ false };
	uint64_t currFenceValue = 0;
};

ActionFitsNvdbRenderer::ActionFitsNvdbRenderer()
{
	renderer_ = std::make_unique<FitsNvdbRenderer>();
}

auto ActionFitsNvdbRenderer::initialize(void* data) -> void
{
	if (data == nullptr)
	{
		return;
	}

	auto device = renderAPI_->getUnityInterfaces()->Get<IUnityGraphicsD3D11>()->GetDevice();


	IDXGIDevice* dxgiDevice{ nullptr };

	auto result = device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice));
	if (result != S_OK)
	{
		logger_->log("Could not query dxgiDevice");
	}

	IDXGIAdapter* dxgiAdapter{ nullptr };
	result = dxgiDevice->GetAdapter(&dxgiAdapter);
	if (result != S_OK)
	{
		logger_->log("Could not get Adapter from dxgiDevice");
	}

	int cudaDevice;
	cudaD3D11GetDevice(&cudaDevice, dxgiAdapter);

	cudaSetDevice(cudaDevice);
	cudaDeviceProp cudaDevProps;
	cudaGetDeviceProperties(&cudaDevProps, cudaDevice);

	const RenderingDataBuffer rdb{ unityDataSchema, 1, data };
	setTextures(rdb);

	renderingDataWrapper_.data.rendererInitializationInfo.deviceUuid = cudaDevProps.uuid;

	const auto colorMapsTexture = rdb.get<UnityTexture>("colorMapsTexture");
	const auto coloringInfo = rdb.get<UnityColoringInfo>("coloringInfo");

	colorMapsTexture_ = renderAPI_->createTexture(colorMapsTexture->texturePointer);
	if (colorMapsTexture_->isValid())
	{
		colorMapsTexture_->registerCUDA(cudaGraphicsRegisterFlagsTextureGather, cudaGraphicsMapFlagsReadOnly);
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
		transferFunctionTexture_->registerCUDA(cudaGraphicsRegisterFlagsTextureGather, cudaGraphicsMapFlagsReadOnly);
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

auto ActionFitsNvdbRenderer::teardown() -> void
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
}

auto ActionFitsNvdbRenderer::customRenderEvent(int eventId, void* data) -> void
{
	logger_->log("Nano custom render event");
	if (isInitialized_ && isReady_ && colorTexture_->isValid() &&
		eventId == static_cast<int>(NanoRenderEventTypes::renderEvent))
	{
		auto device = renderAPI_->getUnityInterfaces()->Get<IUnityGraphicsD3D11>()->GetDevice();


		IDXGIDevice* dxgiDevice{ nullptr };

		auto result = device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice));
		if (result != S_OK)
		{
			logger_->log("Could not query dxgiDevice");
		}

		IDXGIAdapter* dxgiAdapter{ nullptr };
		result = dxgiDevice->GetAdapter(&dxgiAdapter);
		if (result != S_OK)
		{
			logger_->log("Could not get Adapter from dxgiDevice");
		}

		int cudaDevice;
		cudaD3D11GetDevice(&cudaDevice, dxgiAdapter);

		cudaSetDevice(cudaDevice);
		const RenderingDataBuffer rdb{ unityDataSchema, 1, data };

		renderingDataWrapper_.data.renderTargets.colorRt = { .target = colorTexture_->getCudaGraphicsResource(),
															 .extent = {
																 static_cast<uint32_t>(colorTexture_->getWidth()),
																 static_cast<uint32_t>(colorTexture_->getHeight()),
																 static_cast<uint32_t>(colorTexture_->getDepth()) } };

		const auto coloringInfo = rdb.get<UnityColoringInfo>("coloringInfo");
		renderingDataWrapper_.data.coloringInfo = *coloringInfo;


		if (transferFunctionTexture_->isValid())
		{
			renderingDataWrapper_.data.transferFunctionTexture = {
				.target = transferFunctionTexture_->getCudaGraphicsResource(),
				.extent = { static_cast<uint32_t>(transferFunctionTexture_->getWidth()),
							static_cast<uint32_t>(transferFunctionTexture_->getHeight()),
							static_cast<uint32_t>(transferFunctionTexture_->getDepth()) }
			};
		}

		if (colorMapsTexture_->isValid())
		{
			renderingDataWrapper_.data.colorMapTexture = { .target = colorMapsTexture_->getCudaGraphicsResource(),
														   .extent = {
															   static_cast<uint32_t>(colorMapsTexture_->getWidth()),
															   static_cast<uint32_t>(colorMapsTexture_->getHeight()),
															   static_cast<uint32_t>(colorMapsTexture_->getDepth()) } };
		}


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

		const auto& nanovdbLoadingData = rdb.get<UnityNanoVdbLoading>("nanovdbData");

		if (nanovdbLoadingData->newVolumeAvailable)
		{
			const auto pathToNanoVdb = std::filesystem::path{ nanovdbLoadingData->f1 };
			//renderer_->addNanoVdb(pathToNanoVdb);
		}

		// renderer_->selectDataSet(nanovdbLoadingData->selectedDataset);


		currFenceValue += 1;

		renderer_->render();
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

auto ActionFitsNvdbRenderer::setTextures(const RenderingDataBuffer& renderingDataBuffer) -> void
{
	isReady_ = false;
	const auto& unityRenderTargets = renderingDataBuffer.get<UnityRenderTargets>("renderTargets");
	if (unityRenderTargets->colorTexture.textureExtent.depth > 0)
	{
		auto newColorTexture = renderAPI_->createTexture(unityRenderTargets->colorTexture.texturePointer);
		if (newColorTexture->isValid())
		{
			newColorTexture->registerCUDA(cudaGraphicsRegisterFlagsNone, cudaGraphicsMapFlagsWriteDiscard);
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
			newDepthTexture->registerCUDA(cudaGraphicsRegisterFlagsNone, cudaGraphicsMapFlagsWriteDiscard);
			cudaDeviceSynchronize();
			depthTexture_.swap(newDepthTexture);
			cudaDeviceSynchronize();
		}
	}
	isReady_ = true;
}

// This line is crucial and must stay. Replace type with your newly created action type.
EXTERN_CREATE_ACTION(ActionFitsNvdbRenderer)
