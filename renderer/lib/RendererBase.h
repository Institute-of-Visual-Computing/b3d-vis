#pragma once

#include <memory>
#include <owl/common.h>

#include <array>
#include <string>
#include <vector>
#include "DebugDrawListBase.h"
#include "GizmoHelperBase.h"

#include "cuda_runtime.h"

namespace b3d::renderer
{
	struct Camera
	{
		owl::vec3f origin;
		owl::vec3f at;
		owl::vec3f up;
		float cosFoV;
		float FoV;//in radians
		bool directionsAvailable{ false };
		owl::vec3f dir00;
		owl::vec3f dirDu;
		owl::vec3f dirDv;
	};

	struct Extent
	{
		uint32_t width;
		uint32_t height;
		uint32_t depth;
	};
	struct ExternalRenderTarget
	{
		cudaGraphicsResource_t target;
		Extent extent;
	};

	enum class RenderMode : int
	{
		mono = 0,
		stereo
	};

	struct View
	{
		std::array<Camera, 2> cameras;
		RenderMode mode;
		ExternalRenderTarget colorRt;
		ExternalRenderTarget minMaxRt;
		uint64_t fenceValue {0};
	};

	struct VolumeTransform
	{
		owl::vec3f position {0,0,0};
		owl::vec3f scale { 1, 1, 1};
		owl::Quaternion3f rotation;
	};

	struct RendererState
	{
		VolumeTransform volumeTransform {};
	};

	struct RendererInitializationInfo
	{
		cudaExternalSemaphore_t waitSemaphore;
		cudaExternalSemaphore_t signalSemaphore;
		cudaUUID_t deviceUuid;
	};

	struct DebugInitializationInfo
	{
		std::shared_ptr<DebugDrawListBase> debugDrawList{};
		std::shared_ptr<GizmoHelperBase> gizmoHelper{};
	};

	class RendererBase
	{
	public:
		virtual ~RendererBase() = default;

		auto initialize(const RendererInitializationInfo& initializationInfo, const DebugInitializationInfo& debugInitializationInfo) -> void;
		auto deinitialize() -> void;
		auto gui() -> void;
		auto render(const View& view) -> void;

		auto setRenderState(std::unique_ptr<RendererState> newRenderState) -> void
		{
			rendererState_ = std::move(newRenderState);
		}

		[[nodiscard]] auto debugDraw() const -> DebugDrawListBase&
		{
			return *debugInfo_.debugDrawList;
		}

	protected:
		virtual auto onInitialize() -> void{};
		virtual auto onDeinitialize() -> void{};

		virtual auto onGui() -> void{};
		virtual auto onRender(const View& view) -> void = 0;

		RendererInitializationInfo initializationInfo_{};
		DebugInitializationInfo debugInfo_{};

		std::unique_ptr<RendererState> rendererState_{ nullptr };
	};

	auto addRenderer(std::shared_ptr<RendererBase> renderer, const std::string& name) -> void;

	struct RendererRegistryEntry
	{
		std::shared_ptr<RendererBase> rendererInstance;
		std::string name;
	};

	extern std::vector<RendererRegistryEntry> registry;

	template <typename T>
	auto registerRenderer(const std::string& name) -> void
	{
		registry.push_back({ std::make_shared<T>(), name });
	}

	inline auto getRendererIndex(const std::string& name) -> int
	{
		auto index = -1;
		for(auto i = 0; i < registry.size(); i++ )
		{
			if(registry[i].name == name)
			{
				index = i;
				break;
			}
		}
		return index;
	}

} // namespace b3d::renderer
