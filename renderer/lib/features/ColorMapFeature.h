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
		auto beginUpdate() -> void override;
		auto endUpdate() -> void override;
		auto gui() -> void override;

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

		int selectedColoringMode_{0};
		int selectedColoringMap_{0};

	};

} // namespace b3d::renderer
