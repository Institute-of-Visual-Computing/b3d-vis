#pragma once
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "RenderFeature.h"

#include <vector>


namespace b3d::renderer
{
	class TransferFunctionFeature final : public RenderFeature
	{
	public:
		explicit TransferFunctionFeature(const std::string& name, const size_t dataPointsCount = 10);
		auto beginUpdate() -> void override;
		auto endUpdate() -> void override;
		auto gui() -> void override;
		struct ParamsData
		{
			//TODO: put your computed params here
			cudaTextureObject_t transferFunctionTexture{};

		};

		[[nodiscard]] auto getParamsData() -> ParamsData;
	private:
		bool skipUpdate{ false };

		int selectedCurveHandleIdx_{-1};
		std::vector<ImVec2> dataPoints_;
		std::vector<float> stagingBuffer_;
		bool newDataAvailable_{ false };

		ExternalTexture* transferFunctionTexture_;

		cudaArray_t transferFunctionCudaArray_{ nullptr };
		cudaTextureObject_t transferFunctionCudaTexture_{};
	};
} // namespace b3d::renderer
