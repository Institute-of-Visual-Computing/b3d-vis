#pragma once

#include <filesystem>

#include "framework/ModalViewBase.h"

class OpenFileDialogView final : public ModalViewBase
{
public:
	explicit OpenFileDialogView(ApplicationContext& applicationContext,
								  std::function<void(ModalViewBase* self)> onOpenCallback,
								  std::function<void(ModalViewBase* self)> onSubmitCallback);
	~OpenFileDialogView() override;

	struct Model
	{
		std::filesystem::path selectedPath_{};
	};

	auto onDraw() -> void override;

	auto getModel() -> Model;
	auto setViewParams(const std::filesystem::path& currentPath, const std::vector<std::string>& filter) -> void;

private:
	Model model_{};

	std::vector<std::string> filter_{};
	std::filesystem::path currentPath_{};
};
