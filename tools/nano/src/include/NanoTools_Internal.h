#pragma once

#include <nanovdb/GridHandle.h>

#include "NanoResult.h"

namespace b3d::tools::nano
{
	auto extractOffsetSizeToNanoResult(const nanovdb::GridHandle<>& gridHandle, b3d::tools::nano::NanoResult& result) -> void;
}// namespace b3d::tools::nano
