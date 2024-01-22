#pragma once

#include <filesystem>
#include <map>

#include "Common.h"

using ClusterId = int;

auto extractPerClusterBox(const std::filesystem::path& srcFile, const Box3I& searchBox = Box3I::maxBox(), const Vec3I& perBatchSearchSize = Vec3I{})
	-> std::map<ClusterId, Box3I>;
