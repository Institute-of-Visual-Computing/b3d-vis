#pragma once

#include <filesystem>
#include <future>

#include "RenderFeature.h"

#include <vector>

#include <imgui.h>

namespace b3d::renderer
{
	class GizmoHelperBase;
	class DebugDrawListBase;
}

namespace b3d::renderer
{
	class RendererBase;

	struct SoFiaRequest
	{
		owl::box3i regionToSearch;
		std::filesystem::path fitsInputPath;

		// Other Sofia-Params required
	};

	struct SofiaSearchResult
	{
		int errorCode;
		bool wasSuccess;
		std::filesystem::path outputPath;
		std::string message;
	};

	struct SoFiaSearchRequest
	{
		SoFiaRequest request;
		std::string searchHash;
		SofiaSearchResult result;
		
	};

	class SoFiASubregionFeature final : public RenderFeature
	{
	public:
		explicit SoFiASubregionFeature(const std::string& name, b3d::renderer::RendererBase	*renderer);

		auto onInitialize() -> void override;
		
		auto gui() -> void override;

		struct ParamsData
		{
			// TODO: put your computed params here
		};

		[[nodiscard]] auto getParamsData() -> ParamsData;

		[[nodiscard]] auto hasGui() const -> bool override;


	private:

		std::mutex searchMutex_;
		std::atomic_bool stopSearch_{ false };

		RendererBase* renderer_;
		GizmoHelperBase* debugDrawList_;
		owl::AffineSpace3f boxTransform_{};

		std::future<SoFiaSearchRequest> currentSearch;

		std::unique_ptr<SoFiaSearchRequest> lastSearch{ nullptr };


		auto prepareAndExecuteSearch(owl::box3f volumeVoxelBox) -> SoFiaRequest;
		SoFiaSearchRequest search(SoFiaRequest request, std::atomic_bool &stopToken);
	public:
		~SoFiASubregionFeature() override;

	};
} // namespace b3d::renderer
