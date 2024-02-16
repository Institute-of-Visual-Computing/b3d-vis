#pragma once

#include <memory>
#include <owl/common.h>

#include <array>
#include <string>
#include <vector>
#include "DebugDrawListBase.h"
#include "GizmoHelperBase.h"

#include "RenderData.h"


namespace b3d::renderer
{
	struct DebugInitializationInfo
	{
		std::shared_ptr<DebugDrawListBase> debugDrawList{};
		std::shared_ptr<GizmoHelperBase> gizmoHelper{};
	};

	class RendererBase
	{
	public:
		virtual ~RendererBase() = default;

		auto initialize(RenderingDataBuffer* initializationInfo, const DebugInitializationInfo& debugInitializationInfo)
			-> void;
		auto deinitialize() -> void;
		auto gui() -> void;
		auto render() -> void;

		[[nodiscard]] auto debugDraw() const -> DebugDrawListBase&
		{
			return *debugInfo_.debugDrawList;
		}

	protected:
		virtual auto onInitialize() -> void{};
		virtual auto onDeinitialize() -> void{};

		virtual auto onGui() -> void{};
		virtual auto onRender() -> void = 0;

		RenderingDataBuffer* renderData_{};

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
