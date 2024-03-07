#pragma once
#include <nlohmann/json.hpp>

namespace b3d::tools::colormap
{
	struct ColorMap
	{
		std::string colorMapFilePath{};
		std::vector<std::string> colorMapNames{};
		int height;
		int width;
		int pixelsPerMap;
		int colorMapCount;
		
		NLOHMANN_DEFINE_TYPE_INTRUSIVE(ColorMap, colorMapFilePath, colorMapNames, width, height, pixelsPerMap,
									   colorMapCount)

		float firstColorMapYTextureCoordinate;
		float colorMapHeightNormalized;
	};

	auto load(const std::filesystem::path& file) -> ColorMap;
}
