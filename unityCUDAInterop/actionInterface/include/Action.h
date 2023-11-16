#pragma once

#include "IUnityInterface.h"
#include "RenderAPI.h"

namespace b3d::unity_cuda_interop
{
	

	class PluginLogger;
	class Action
	{
	public:
		Action() = default;
		virtual ~Action() = default;

		auto renderEventAndData(const int eventID, void* data) -> void
		{
			if (eventID - renderEventIDOffset_ == 0)
			{
				initialize(data);
			}
			else if (eventID - renderEventIDOffset_ == 1)
			{
				teardown();
			}
			else
			{
				customRenderEvent(eventID, data);
			}
		}

		virtual auto getRenderEventIDCount() -> int { return renderEventIDCount_; }
		virtual auto getRenderEventIDOffset() const -> int { return renderEventIDOffset_; }
		virtual auto getAboveValidRenderEventID() -> int { return aboveValidRenderEventID_; }

		virtual auto isValidEventID(const int eventID) -> bool
		{ return eventID < aboveValidRenderEventID_ && eventID >= renderEventIDOffset_; }

		auto runtimeInitialize(const int renderEventIdOffset, PluginLogger* logger, runtime::RenderAPI* renderAPI) -> void
		{
			renderEventIDOffset_ = renderEventIdOffset;
			aboveValidRenderEventID_ = renderEventIdOffset + renderEventIDCount_;
			logger_ = logger;
			renderAPI_ = renderAPI;
			isRegistered_ = true;
		}

		auto runtimeTearDown() -> void
		{

			logger_ = nullptr;
			renderAPI_ = nullptr;
			isRegistered_ = false;
		}

	protected:
		virtual auto initialize(void* data) -> void = 0;
		virtual auto teardown() -> void = 0;
		virtual auto customRenderEvent(int eventId, void* data) -> void = 0;

		bool isRegistered_{ false };

		int renderEventIDOffset_{ 0 };
		int renderEventIDCount_{ 0 };
		int aboveValidRenderEventID_{ 0 };

		PluginLogger* logger_{ nullptr };
		runtime::RenderAPI* renderAPI_{ nullptr };
	};

} // namespace b3d::unity_cuda_interop

extern "C"
{
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API createAction() -> b3d::unity_cuda_interop::Action*;
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API getRenderEventIDOffset(const b3d::unity_cuda_interop::Action* nativeAction) -> int;
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API destroyAction(b3d::unity_cuda_interop::Action* nativeAction) -> void;
}
