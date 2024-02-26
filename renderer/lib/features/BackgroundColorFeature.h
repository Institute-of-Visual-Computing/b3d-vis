#pragma once

#include <RenderData.h>
#include "RenderFeature.h"

namespace b3d::renderer
{
	// template <unsigned int S>
	class BackgroundColorFeature final : public RenderFeature
	{
	public:
		explicit BackgroundColorFeature(const std::string& name) : RenderFeature{ name }
		{
		}

		explicit BackgroundColorFeature(const std::string& name, std::array<ColorRGB, 2> colors)
			: RenderFeature{ name }, colors_{ colors }
		{
		}

		auto gui() -> void override;
		[[nodiscard]] auto hasGui() const -> bool override;

		struct ParamsData
		{
			std::array<ColorRGB, 2> colors;
		};

		[[nodiscard]] auto getParamsData() -> ParamsData;

	private:
		std::array<ColorRGB, 2> colors_ {};
	};

} // namespace b3d::renderer
