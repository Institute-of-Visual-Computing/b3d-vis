#pragma once

#include <cstdint>

#include "Flags.h"

enum class GizmoOperationFlagBits : uint16_t
{
	none = 0,
	translate = 1,
	rotate = 2,
	scale = 4
};

using GizmoOperationFlags = Flags<GizmoOperationFlagBits>;

inline GizmoOperationFlags operator|(const GizmoOperationFlagBits& a, const GizmoOperationFlagBits& b)
{
	auto flags = GizmoOperationFlags{ a };
	flags |= b;
	return flags;
}
