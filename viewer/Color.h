#pragma once

#include <cstdint>
#include <imgui.h>

#include "Mathematics.h"

#ifdef WIN32
#include <winrt/Windows.UI.ViewManagement.h>
#endif

struct Color
{
public:
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;

public:
	constexpr Color(float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 1.0f)
		: r{ static_cast<uint8_t>(r * 255) }, g{ static_cast<uint8_t>(g * 255) }, b{ static_cast<uint8_t>(b * 255) },
		  a{ static_cast<uint8_t>(a * 255) }
	{
	}

	constexpr Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255u) : r{ r }, g{ g }, b{ b }, a{ a }
	{
	}

	// constexpr Color(const Vector4& v) : r_{v.x}, g_{v.y}, b_{v.z}, a_{v.w}
	//{
	// }
#if WIN32
	constexpr Color(const winrt::Windows::UI::Color& color) : r{ color.R }, g{ color.G }, b{ color.B }, a{ color.A }
	{
	}
#endif
	constexpr operator ImColor() const
	{
		return ImColor{ static_cast<int>(r), static_cast<int>(g), static_cast<int>(b), static_cast<int>(a) };
	}

	constexpr operator ImU32() const
	{
		return IM_COL32(r, g, b, a);
	}

	explicit operator ImVec4() const
	{
		return ImGui::ColorConvertU32ToFloat4(IM_COL32(r, g, b, a));
	}
};
