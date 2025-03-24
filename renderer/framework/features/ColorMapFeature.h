#pragma once

#include <RenderData.h>
#include "RenderFeature.h"

namespace b3d::renderer
{

	class ColorMapFeature final : public RenderFeature
	{
	public:
		explicit ColorMapFeature(const std::string& name) : RenderFeature{ name }
		{
		}
		auto onInitialize() -> void override;
		auto beginUpdate() -> void override;
		auto endUpdate() -> void override;

		struct ParamsData
		{
			cudaTextureObject_t colorMapTexture;
			owl::vec4f uniformColor;
			float selectedColorMap;
			ColoringMode mode;
		};

		[[nodiscard]] auto getParamsData() -> ParamsData;
	private:
		bool skipUpdate{false};
		ColoringInfo* coloringInfo_{nullptr};
		ColorMapInfos* colorMapInfos_{nullptr};
		ExternalTexture* colorMapTexture_{nullptr};
		cudaTextureObject_t colorMapCudaTexture_{};
	};

} // namespace b3d::renderer
