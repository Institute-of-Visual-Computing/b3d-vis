#pragma once

#include <memory>
#include <owl/common.h>

#include <array>
#include <string>
#include <vector>
#include "DebugDrawListBase.h"

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


	struct RendererInitializationInfo
	{
		cudaExternalSemaphore_t waitSemaphore;
		cudaExternalSemaphore_t signalSemaphore;
		cudaUUID_t deviceUuid;
	};

	struct DebugInitializationInfo
	{
		std::shared_ptr<DebugDrawListBase> debugDrawList{};
	};

	class RendererBase
	{
	public:
		virtual ~RendererBase() = default;

		auto initialize(const RendererInitializationInfo& initializationInfo, const DebugInitializationInfo& debugInitializationInfo) -> void;
		auto deinitialize() -> void;
		auto gui() -> void;
		auto render(const View& view) -> void;

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
