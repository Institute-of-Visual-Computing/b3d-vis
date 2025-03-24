#pragma once

#pragma once

#include "RenderData.h"
#include "RenderFeature.h"

namespace b3d::renderer
{

	class RenderTargetFeature final : public RenderFeature
	{
	public:
		explicit RenderTargetFeature(const std::string& name) : RenderFeature{ name }
		{
		}

		auto beginUpdate() -> void override;
		auto endUpdate() -> void override;

		struct ParamsData
		{
			CudaStereoRenderTarget colorRT;
			CudaStereoRenderTarget minMaxRT;
		};

		[[nodiscard]] auto getParamsData() -> ParamsData;

	private:
		bool skipUpdate{ false };
		
		RenderTargets* renderTargets_{ nullptr };
		CudaStereoRenderTarget cudaColorRT_{};
		CudaStereoRenderTarget cudaMinMaxRT_{};
	};

} // namespace b3d::renderer
