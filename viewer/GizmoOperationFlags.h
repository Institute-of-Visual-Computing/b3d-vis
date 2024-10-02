#pragma once

#include <cstdint>
#include <type_traits>

#include "Flags.h"

enum class GizmoOperationFlagBits : uint16_t
{
	none = 0,
	translate = 1,
	rotate = 2,
	scale = 4
};

using GizmoOperationFlags = Flags<GizmoOperationFlagBits>;
