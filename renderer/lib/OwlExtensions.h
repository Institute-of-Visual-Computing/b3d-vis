#pragma once

#include <owl/common/math/LinearSpace.h>
#include <owl/common/math/vec.h>

#include <algorithm>

namespace owl::extensions
{
	inline auto owlParamsSet4f(OWLParams obj, const char* name, const vec4f& val) -> void
	{
		owlParamsSet4f(obj, name, val.x, val.y, val.z, val.w);
	}

	inline auto owlParamsSet3f(OWLParams obj, const char* name, const vec3f& val) -> void
	{
		owlParamsSet3f(obj, name, val.x, val.y, val.z);
	}

	inline auto owlParamsSet3f(OWLParams obj, const char* name, const b3d::renderer::ColorRGB& val) -> void
	{
		owlParamsSet3f(obj, name, val.r, val.g, val.b);
	}

	inline auto owlMissProgSet3f(OWLMissProg obj, const char* name, const b3d::renderer::ColorRGB& val) -> void
	{
		owlMissProgSet3f(obj, name, val.r, val.g, val.b);
	}

	template <typename T>
	static auto scale(const owl::LinearSpace3<T>& a) -> typename T
	{
		return T(length(a.vx), length(a.vy), length(a.vz));
	}

	template <typename T>
	static auto rotation(const LinearSpace3<T>& a) -> QuaternionT<typename T::scalar_t>
	{
		const auto absQ = std::cbrt(a.det());

		using scalar = typename T::scalar_t;

		auto q = QuaternionT<scalar>{};
		q.r = sqrt(std::max<scalar>({ 0, absQ + a.vx.x + a.vy.y + a.vz.z })) / 2;
		q.i = sqrt(std::max<scalar>({ 0, absQ + a.vx.x - a.vy.y - a.vz.z })) / 2;
		q.j = sqrt(std::max<scalar>({ 0, absQ - a.vx.x + a.vy.y - a.vz.z })) / 2;
		q.k = sqrt(std::max<scalar>({ 0, absQ - a.vx.x - a.vy.y + a.vz.z })) / 2;

		q.i = std::copysign(q.i, a.vz.y - a.vy.z);
		q.j = std::copysign(q.j, a.vx.z - a.vz.x);
		q.k = std::copysign(q.k, a.vy.x - a.vx.y);

		return q;
	}
} // namespace owl::extensions
