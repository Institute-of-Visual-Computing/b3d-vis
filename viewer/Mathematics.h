#pragma once

#include <imgui.h>

struct Vector2
{
public:
	float x{ 0.0f };
	float y{ 0.0f };

public:
	constexpr Vector2()
	{
	}

	constexpr Vector2(float x, float y) : x{ x }, y{ y }
	{
	}

	constexpr explicit Vector2(float s) : x{ s }, y{ s }
	{
	}

	constexpr Vector2(const ImVec2& v) : x{ v.x }, y{ v.y }
	{
	}

	constexpr operator ImVec2() const
	{
		return ImVec2{ x, y };
	}

	friend inline constexpr auto operator*(const Vector2& lhs, const float s) -> Vector2
	{
		return Vector2{ lhs.x * s, lhs.y * s };
	}

	friend inline constexpr auto operator*(const float s, const Vector2& rhs) -> Vector2
	{
		return Vector2{ rhs.x * s, rhs.y * s };
	}

	constexpr auto operator*=(const float s) -> Vector2&
	{
		x *= s;
		y *= s;
		return *this;
	}
};

struct Vector4
{
public:
	float x{ 0.0f };
	float y{ 0.0f };
	float z{ 0.0f };
	float w{ 0.0f };

public:
	constexpr Vector4()
	{
	}

	constexpr Vector4(float x, float y, float z, float w) : x{ x }, y{ y }, z{ z }, w{ w }
	{
	}

	constexpr Vector4(float s) : x{ s }, y{ s }, z{ s }, w{ s }
	{
	}

	constexpr Vector4(const ImVec4& v) : x{ v.x }, y{ v.y }, z{ v.z }, w{ v.w }
	{
	}

	constexpr operator ImVec4() const
	{
		return ImVec4{ x, y, z, w };
	}

	friend inline constexpr auto operator*(const Vector4& lhs, const float s) -> Vector4
	{
		return Vector4{ lhs.x * s, lhs.y * s, lhs.z * s, lhs.w * s };
	}

	friend inline constexpr auto operator*(const float s, const Vector4& rhs) -> Vector4
	{
		return Vector4{ rhs.x * s, rhs.y * s, rhs.z * s, rhs.w * s };
	}

	constexpr auto operator*=(const float s) -> Vector4&
	{
		x *= s;
		y *= s;
		z *= s;
		w *= s;
		return *this;
	}
};
