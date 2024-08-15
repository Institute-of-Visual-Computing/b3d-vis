#pragma once

#include <corecrt_math_defines.h>
#include <functional>
#include <vector>

namespace animation
{

	inline auto fastAtan(const float x) -> float
	{
		const auto z = fabs(x);
		const auto w = z > 1.0f ? 1.0f / z : z;
		const float y = (M_PI / 4.0f) * w - w * (w - 1) * (0.2447f + 0.0663f * w);
		return copysign(z > 1.0f ? M_PI / 2.0 - y : y, x);
	}

	inline auto fastNegExp(const float x) -> float
	{
		return 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
	}

	inline auto squaref(const float x) -> float
	{
		return x * x;
	}

	inline auto springDamperExact(float& x, float& v, const float xGoal, const float vGoal, const float stiffness,
						   const float damping, const float dt, const float eps = 1e-5f) -> void
	{
		const auto g = xGoal;
		const auto q = vGoal;
		const auto s = stiffness;
		const auto d = damping;
		const auto c = g + (d * q) / (s + eps);
		const auto y = d / 2.0f;
		const auto w = sqrtf(s - (d * d) / 4.0f);
		auto j = sqrtf(squaref(v + y * (x - c)) / (w * w + eps) + squaref(x - c));
		const auto p = fastAtan((v + (x - c) * y) / (-(x - c) * w + eps));

		j = (x - c) > 0.0f ? j : -j;

		const auto eydt = fastNegExp(y * dt);

		x = j * eydt * cosf(w * dt + p) + c;
		v = -y * j * eydt * cosf(w * dt + p) - w * j * eydt * sinf(w * dt + p);
	}


	using PropertyAnimation = std::function<void(float, float)>;

	class PropertyAnimator final
	{
	public:
		auto addPropertyAnimation(const PropertyAnimation& animation)
		{
			animations_.push_back(animation);
		}

		auto animate(const float delta) -> void
		{
			if (isRunning_)
			{
				globalTime_ += delta;
				for (auto& animation : animations_)
				{
					animation(globalTime_, delta);
				}
			}
		}

		auto reset() -> void
		{
			animations_.clear();
		}
		auto start() -> void
		{
			isRunning_ = true;
		}
		auto stop() -> void
		{
			pause();
			globalTime_ = 0.0f;
		}
		auto pause() -> void
		{
			isRunning_ = false;
		}

		auto isRunning() const -> bool
		{
			return isRunning_;
		}

	private:
		std::vector<PropertyAnimation> animations_{};
		float globalTime_{ 0.0f };

		bool isRunning_{ false };
	};
} // namespace animation
