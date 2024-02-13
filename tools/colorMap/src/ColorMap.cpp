#include "ColorMap.h"

#include <fstream>
#include <iostream>

using namespace b3d::tools;

auto colormap::load(const std::filesystem::path& file) -> ColorMap
{
	assert(std::filesystem::exists(file));
	std::ifstream f(file);

	auto colorMap = ColorMap{};

	try
	{
		const auto data = nlohmann::json::parse(f);
		colorMap = data.get<ColorMap>();
	}
	catch (nlohmann::json::type_error& e)
	{
		std::cout << e.what();
		// [json.exception.type_error.304] cannot use at() with object
	}

	colorMap.colorMapHeightNormalized = (1.0f / static_cast<float>(colorMap.height)) * colorMap.pixelsPerMap;
	colorMap.firstColorMapYTextureCoordinate = colorMap.colorMapHeightNormalized / 2.0f;


	return colorMap;
}
