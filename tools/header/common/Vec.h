#pragma once

#include <algorithm>

#ifdef B3D_USE_NLOHMANN_JSON
#include "nlohmann/json.hpp"
#endif

namespace b3d::common
{
	template <typename T>
	struct Vec3
	{
		T x{};
		T y{};
		T z{};

		inline auto clamp(const Vec3<T>& min, const Vec3<T>& max) -> void
		{
			x = std::clamp(x, min.x, max.x);
			y = std::clamp(y, min.y, max.y);
			z = std::clamp(z, min.z, max.z);
		}
		auto operator<=>(const Vec3<T>& other) const -> auto = default;
	};

	template <typename T>
	inline auto clamp(const Vec3<T>& value, const Vec3<T>& min, const Vec3<T>& max) -> Vec3<T>
	{
		Vec3<T> r = value;
		r.clamp(min, max);
		return r;
	}

	template <typename T>
	[[nodiscard]] auto min(const Vec3<T>& v1, const Vec3<T>& v2) -> Vec3<T>
	{
		Vec3<T> r;
		r.x = std::min(v1.x, v2.x);
		r.y = std::min(v1.y, v2.y);
		r.z = std::min(v1.z, v2.z);
		return r;
	}

	template <typename T>
	[[nodiscard]] auto max(const Vec3<T>& v1, const Vec3<T>& v2) -> Vec3<T>
	{
		Vec3<T> r;
		r.x = std::max(v1.x, v2.x);
		r.y = std::max(v1.y, v2.y);
		r.z = std::max(v1.z, v2.z);
		return r;
	}

	using Vec3F = Vec3<float>;
	using Vec3d = Vec3<double>;
	using Vec3I = Vec3<int>;
	using Vec3i64 = Vec3<int64_t>;

	/// \brief Returns the index of a 3D array when flattened.
	/// \param axisSize Size of the array in each dimension.
	/// \param x Fastest running index. Must be smaller than axisSize.x
	/// \param y Second fastest running index. Must be smaller than axisSize.y
	/// \param z slowest running index. Must be smaller than axisSize.z
	/// \return Index of the 3D array when flattened.
	[[nodiscard]] inline auto flattenIndex(const Vec3I axisSize, const uint64_t x, const uint64_t y, const uint64_t z)
		-> uint64_t
	{
		return static_cast<uint64_t>(axisSize.x) * static_cast<uint64_t>(axisSize.y) * z +
			static_cast<uint64_t>(axisSize.x) * y + x;
	}

	/// \brief Returns the index of a 3D array when flattened.
	/// \param axisSize Size of the array in each dimension.
	///	\param coordinate 3D coordinate.
	/// \return Index of the 3D array when flattened.
	[[nodiscard]] inline auto flattenIndex(const Vec3I& axisSize, const Vec3I& coordinate) -> uint64_t
	{
		return flattenIndex(axisSize, coordinate.x, coordinate.y, axisSize.z - coordinate.z - 1);
	}

#ifdef B3D_USE_NLOHMANN_JSON
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Vec3F, x, y, z);
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Vec3d, x, y, z);
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Vec3I, x, y, z);
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Vec3i64, x, y, z);
#endif

} // namespace b3d::common
