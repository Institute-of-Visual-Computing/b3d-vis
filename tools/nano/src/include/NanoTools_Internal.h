#pragma once

#include <nanovdb/util/GridHandle.h>

#include "NanoResult.h"

namespace b3d::tools::nano
{
	auto extractOffsetSizeToNanoResult(const nanovdb::GridHandle<>& gridHandle, b3d::tools::nano::NanoResult& result) -> void;
}// namespace b3d::tools::nano
