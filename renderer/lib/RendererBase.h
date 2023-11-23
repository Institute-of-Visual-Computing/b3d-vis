#pragma once

#include <memory>
#include <owl/common.h>

#include <string>
#include <vector>
#include "cuda_runtime.h"

namespace b3d::renderer
{
	struct Camera
	{
		owl::vec3f origin;
		owl::vec3f at;
		owl::vec3f up;
		float cosFoV;
	};

	struct View
	{
		Camera camera1;
	};

	enum class RenderMode
{
	mono,
	stereo
};

struct RendererInitializationInfo
{
	// on stereo we expect that those resources are of a array type in native API
	cudaGraphicsResource_t colorRt;
	cudaGraphicsResource_t minMaxRt;
	RenderMode mode{ RenderMode::mono };

	cudaExternalSemaphore_t waitSemaphore;
	cudaExternalSemaphore_t signalSemaphore;

	cudaUUID_t deviceUuid;
};

	class RendererBase
	{
	public:
		virtual ~RendererBase() = default;

		auto initialize(const RendererInitializationInfo& initializationInfo) -> void;
		auto deinitialize() -> void;
		auto gui() -> void;
		auto render(const View& view) -> void;

	protected:
		virtual auto onInitialize() -> void{};
		virtual auto onDeinitialize() -> void{};

		virtual auto onGui() -> void{};
		virtual auto onRender(const View& view) -> void = 0;

		RendererInitializationInfo initializationInfo_{};
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

} // namespace b3d::renderer
