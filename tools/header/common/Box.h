#pragma once

#include "Vec.h"

namespace b3d::common
{
	template <typename T>
	struct Box3
	{
		Vec3<T> lower{};
		Vec3<T> upper{};

		[[nodiscard]] inline static auto maxBox() -> Box3<T>
		{
			Box3<T> r;
			r.upper.x = std::numeric_limits<T>::max();
			r.upper.y = std::numeric_limits<T>::max();
			r.upper.z = std::numeric_limits<T>::max();

			r.lower.x = std::numeric_limits<T>::min();
			r.lower.y = std::numeric_limits<T>::min();
			r.lower.z = std::numeric_limits<T>::min();
			return r;
		}

		inline auto extend(const Vec3<T>& point) -> void
		{
			lower = min(lower, point);
			upper = max(upper, point);
		}

		inline auto extend(const Box3<T>& box) -> void
		{
			extend(box.lower);
			extend(box.upper);
		}

		inline auto clip(const Box3<T>& clipBox)
		{
			lower = clamp(lower, clipBox.lower, clipBox.upper);
			upper = clamp(upper, clipBox.lower, clipBox.upper);
		}

		[[nodiscard]] inline auto size() const -> Vec3<T>
		{
			Vec3<T> r;
			r.x = std::abs(upper.x - lower.x);
			r.y = std::abs(upper.y - lower.y);
			r.z = std::abs(upper.z - lower.z);
			return r;
		}

		[[nodiscard]] inline auto volume() const -> Vec3<T>
		{
			Vec3<T> s = size();
			return s.x * s.y * s.z;
		}

		[[nodiscard]] inline auto contains(const Vec3<T>& value) const -> bool
		{
			return (value >= lower) && (value <= upper);
		}

		auto operator<=>(const Box3<T>& other) const -> auto = default;
	};

	template <typename T>
	[[nodiscard]] auto operator*(const Vec3<T>& a, const Vec3<T>& b) -> Vec3<T>
	{
		Vec3<T> r;
		r.x = a.x * b.x;
		r.y = a.y * b.y;
		r.z = a.z * b.z;
		return r;
	}

	template <typename T1, typename T2>
	[[nodiscard]] auto operator*(const T2& a, const Vec3<T1>& b)
		-> std::enable_if_t<std::is_convertible_v<T1, T2>, Vec3<T1>>
	{
		Vec3<T1> r;
		r.x = a * b.x;
		r.y = a * b.y;
		r.z = a * b.z;
		return r;
	}

	template <typename T1, typename T2>
	[[nodiscard]] auto operator*(const Vec3<T1>& b, const T2& a)
		-> std::enable_if_t<std::is_convertible_v<T1, T2>, Vec3<T1>>
	{
		return a * b;
	}

	template <typename T>
	[[nodiscard]] auto operator+(const Vec3<T>& a, const Vec3<T>& b) -> Vec3<T>
	{
		Vec3<T> r;
		r.x = a.x + b.x;
		r.y = a.y + b.y;
		r.z = a.z + b.z;
		return r;
	}

	template <typename T>
	[[nodiscard]] auto operator-(const Vec3<T>& a, const Vec3<T>& b) -> Vec3<T>
	{
		Vec3<T> r;
		r.x = a.x - b.x;
		r.y = a.y - b.y;
		r.z = a.z - b.z;
		return r;
	}

	template <typename T>
	[[nodiscard]] auto clip(const Box3<T>& value, const Box3<T>& clipBox) -> Box3<T>
	{
		Box3<T> r(value);
		r.clip(clipBox);
		return r;
	}

	using Box3F = Box3<float>;
	using Box3d = Box3<double>;
	using Box3I = Box3<int>;
	using Box3i64 = Box3<int64_t>;

	#ifdef B3D_USE_NLOHMANN_JSON
		NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Box3F, lower, upper);
		NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Box3d, lower, upper);
		NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Box3I, lower, upper);
		NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Box3i64, lower, upper);
	#endif
} // namespace b3d::common
