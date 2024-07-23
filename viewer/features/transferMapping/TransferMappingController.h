#pragma once
#include <memory>
#include "TransferMappingView.h"
#include "framework/UpdatableComponentBase.h"
#include <Common.h>

class TransferMapping;

class TransferMappingController final : public UpdatableComponentBase
{
public:
	TransferMappingController(ApplicationContext& applicationContext, TransferMapping& transferMapping);

	auto update() -> void override;

	struct Model
	{
		b3d::renderer::ColoringMode coloringMode;
		int selectedColorMap{ 0 };
		cudaGraphicsResource_t transferFunctionGraphicsResource{};
		int transferFunctionSamplesCount{};
	};

	auto updateModel(Model& model) const -> bool;

private:
	std::unique_ptr<TransferMappingView> mappingView_{};

	bool showToolWindow_{ true };
	TransferMapping* transferMapping_{};
};
