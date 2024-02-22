#pragma once

#include "IUnityInterface.h"
#pragma once

#include "PluginHandler.h"
// 	TODO: Only required if we're an action which is using RendererBase. Introduce RenderAction and leave this Action as generic as possible.
#include "RendererBase.h"

extern b3d::unity_cuda_interop::PluginHandler sPluginHandler;

namespace b3d::unity_cuda_interop
{
	using UnityCamera = renderer::Camera;
	using UnityExtent = renderer::Extent;
	using UnityRenderMode = renderer::RenderMode;
	using UnityView = renderer::View;
	using UnityColoringMode = renderer::ColoringMode;
	using UnityColoringInfo = renderer::ColoringInfo;

	struct UnityTexture
	{
		void* texturePointer;
		UnityExtent textureExtent;
	};

	struct UnityRenderTargets
	{
		UnityTexture colorTexture;
		UnityTexture depthTexture;
	};

	struct UnityVolumeTransform
	{
		owl::vec3f position{ 0, 0, 0 };
		owl::vec3f scale{ 1, 1, 1 };
		owl::Quaternion3f rotation{ 1 };
	};

	struct UnityRenderingData
	{
		UnityRenderTargets renderTargets;
		UnityView view;
		UnityVolumeTransform volumeTransform;
		UnityTexture colorMapsTexture;
		UnityColoringInfo coloringInfo;			
	};

	static const renderer::SchemaData unityDataSchema { {
			SCHEMA_ENTRY("renderTargets", renderTargets, UnityRenderingData),
			SCHEMA_ENTRY("view", view, UnityRenderingData),
			SCHEMA_ENTRY("volumeTransform", volumeTransform, UnityRenderingData),
			SCHEMA_ENTRY("colorMapsTexture", colorMapsTexture, UnityRenderingData),
			SCHEMA_ENTRY("coloringInfo", coloringInfo, UnityRenderingData),
										  },
		sizeof(UnityRenderingData)
	};

	class PluginLogger;

	class Action
	{
	public:
		static constexpr int eventIdCount = 10;

		Action() = default;

		virtual ~Action()
		{
			isRegistered_ = false;
			logger_ = nullptr;
			renderAPI_ = nullptr;
			renderEventIDOffset_ = -1;
		}

		auto renderEventAndData(const int eventID, void* data) -> void
		{
			const auto actionRenderId = eventID - renderEventIDOffset_;
			customRenderEvent(actionRenderId, data);
		}

		virtual auto getRenderEventIDOffset() const -> int
		{
			return renderEventIDOffset_;
		}

		auto registerAction(const int renderEventIdOffset, PluginLogger* logger, RenderAPI* renderAPI)
		{
			if (isRegistered_)
			{
				return;
			}
			
			renderEventIDOffset_ = renderEventIdOffset;
			logger_ = logger;
			renderAPI_ = renderAPI;

			isRegistered_ = true;
		}

		auto containsEventId(const int eventId) -> bool
		{
			return renderEventIDOffset_ <= eventId && eventId < renderEventIDOffset_ + Action::eventIdCount;
		}

		auto unregisterAction()
		{
			if (!isRegistered_)
			{
				return;
			}

			isRegistered_ = false;

			logger_ = nullptr;
			renderAPI_ = nullptr;
			renderEventIDOffset_ = 0;
		}


		virtual auto initialize(void* data) -> void = 0;
		virtual auto teardown() -> void = 0;

	protected:
		virtual auto customRenderEvent(int eventId, void* data) -> void = 0;
		
		PluginLogger* logger_{ nullptr };
		RenderAPI* renderAPI_{ nullptr };

		renderer::RenderingDataWrapper renderingDataWrapper_{};

		int renderEventIDOffset_{ -1 };
		bool isRegistered_{ false };
		bool isInitialized_{ false };
	};

} // namespace b3d::unity_cuda_interop

extern "C"
{
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API createAction() -> b3d::unity_cuda_interop::Action*;

	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API destroyAction(b3d::unity_cuda_interop::Action* action) -> void;

	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API initializeAction(b3d::unity_cuda_interop::Action* action, void* data) -> void;

	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API teardownAction(b3d::unity_cuda_interop::Action* action) -> void;

	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API getRenderEventIDOffset(const b3d::unity_cuda_interop::Action* action) -> int;
}
