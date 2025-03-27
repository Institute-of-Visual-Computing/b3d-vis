#pragma once

#include "framework/ModalViewBase.h"

#include <memory>

#include "ServerClient.h"

struct NewProjectModel
{
	std::filesystem::path sourcePath;
	std::string projectName;
};

class AddNewProjectView final : public ModalViewBase
{
public:
	AddNewProjectView(ApplicationContext& appContext, const std::string_view name,
					  const std::function<void(ModalViewBase*)>& onSubmitCallback);

	// Inherited via ModalViewBase
	auto onDraw() -> void override;

private:
	NewProjectModel model_{};

public:
	auto setModel(const NewProjectModel& model) -> void
	{
		model_ = model;
	}

	[[nodiscard]] auto model() const -> const NewProjectModel&
	{
		return model_;
	}
};
