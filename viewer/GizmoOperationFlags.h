#pragma once

#include <cstdint>

enum class GizmoOperationFlagBits
{
	none = 0,
	translate = 1,
	rotate = 2,
	scale = 4
};

using GizmoOperationFlags = uint16_t;
