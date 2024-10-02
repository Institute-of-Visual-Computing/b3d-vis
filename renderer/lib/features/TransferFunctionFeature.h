#pragma once

#include "RenderFeature.h"
#include <vector>

namespace b3d::renderer
{
	class TransferFunctionFeature final : public RenderFeature
	{
	public:
		explicit TransferFunctionFeature(const std::string& name, const size_t dataPointsCount = 32);

		auto onInitialize() -> void override;
		auto beginUpdate() -> void override;
		auto endUpdate() -> void override;
		struct ParamsData
		{
			cudaTextureObject_t transferFunctionTexture{};
		};

		[[nodiscard]] auto getParamsData() -> ParamsData;

	private:
		bool skipUpdate{ false };

		ExternalTexture* transferFunctionTexture_;
		cudaArray_t transferFunctionCudaArray_{ nullptr };
		cudaTextureObject_t transferFunctionCudaTexture_{};
	};
} // namespace b3d::renderer
