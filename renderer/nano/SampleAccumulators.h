#pragma once

#include <owl/common/math/vec.h>
#include "SamplerMapper.h"

namespace b3d
{
	namespace renderer
	{
		namespace nano
		{
			class SampleAccumulator
			{
			public:
				virtual ~SampleAccumulator() = default;
				__host__ __device__ virtual auto preAccumulate() -> void
				{
				}
				__host__ __device__ virtual auto postAccumulate() -> void
				{
				}
				__host__ __device__ virtual auto accumulate(const float& value) -> void = 0;

				__host__ __device__ auto getAccumulator() const -> owl::vec4f
				{
					return owl::vec4f{ colorAccumulator_, opacityAccumulator_ };
				}
			protected:
				owl::vec3f colorAccumulator_{};
				float opacityAccumulator_{};
			};

			class AverageIntensityProjection final : public SampleAccumulator
			{
			public:
				__host__ __device__ auto accumulate(const float& value) -> void override
				{
					totalSamples_++;
					const auto weightValue = transferMap(value);
					opacityAccumulator_ = opacityAccumulator_ + (weightValue - opacityAccumulator_) / static_cast<float>(totalSamples_);
				}
				__host__ __device__ auto preAccumulate() -> void override
				{
					totalSamples_ = 0;
					colorAccumulator_ = owl::vec3f{ 0.0f,0.0f,0.0f };
					opacityAccumulator_ = 0.0f;
				}
				__host__ __device__ auto postAccumulate() -> void override
				{
					colorAccumulator_ = colorMap(opacityAccumulator_);
				}
			private:
				int totalSamples_{};
			};

			class MaximumIntensityProjection final : public SampleAccumulator
			{
			public:
				__host__ __device__ auto accumulate(const float& value) -> void override
				{
					const auto weightValue = transferMap(value);
					opacityAccumulator_ = owl::max(opacityAccumulator_, weightValue);
				}
				__host__ __device__ auto preAccumulate() -> void override
				{
					colorAccumulator_ = owl::vec3f{ 0.0f,0.0f,0.0f };
					opacityAccumulator_ = 0.0f;
				}
				__host__ __device__ auto postAccumulate() -> void override
				{
					colorAccumulator_ = colorMap(opacityAccumulator_);
				}
			};

			class IntensityIntegration final : public SampleAccumulator
			{
			public:
				__host__ __device__ auto accumulate(const float& value) -> void override
				{
					const auto colorValue = colorMap(value);
					const auto opacityValue = transferMap(value);
					const auto weight = opacityValue - opacityAccumulator_ * opacityValue;
					colorAccumulator_ += weight * colorValue;
					opacityAccumulator_ += weight;
				}
			};
		}
	}
}
