#pragma once

#include <filesystem>
#include <map>

#include "FitsCommon.h"

namespace  b3d::tools::fitstools
{
	auto getFitsFileInfos(const std::filesystem::path& file, const uint8_t hduIndex = 0)
		-> b3d::tools::common::fits::FitsProperties;
}
