#pragma once

#include <cstdint>
#include <imgui.h>

#include "Mathematics.h"

#ifdef WIN32
#include <winrt/Windows.UI.ViewManagement.h>
#endif

struct Color
{
private:
	uint8_t r_;
	uint8_t g_;
	uint8_t b_;
	uint8_t a_;

public:
	constexpr Color(float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 1.0f)
		: r_{ static_cast<uint8_t>(r * 255) }, g_{ static_cast<uint8_t>(g * 255) }, b_{ static_cast<uint8_t>(b * 255) },
		  a_{ static_cast<uint8_t>(a * 255) }
	{
	}

	constexpr Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255u) : r_{ r }, g_{ g }, b_{ b }, a_{ a }
	{
	}

	// constexpr Color(const Vector4& v) : r_{v.x}, g_{v.y}, b_{v.z}, a_{v.w}
	//{
	// }
#if WIN32
	constexpr Color(const winrt::Windows::UI::Color& color) : r_{ color.R }, g_{ color.G }, b_{ color.B }, a_{ color.A }
	{
	}
#endif
	constexpr operator ImColor() const
	{
		return ImColor{ static_cast<int>(r_), static_cast<int>(g_), static_cast<int>(b_), static_cast<int>(a_) };
	}

	constexpr operator ImU32() const
	{
		return IM_COL32(r_, g_, b_, a_);
	}

	explicit operator ImVec4() const
	{
		return ImGui::ColorConvertU32ToFloat4(IM_COL32(r_, g_, b_, a_));
	}
};
