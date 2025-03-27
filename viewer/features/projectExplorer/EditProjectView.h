#pragma once

#include "framework/ModalViewBase.h"

struct EditProjectModel
{
	std::string projectName;
};

class EditProjectView final : public ModalViewBase
{
public:
	EditProjectView(ApplicationContext& appContext, const std::string_view name,
					std::function<void(ModalViewBase* self)> onOpenCallback,
					std::function<void(ModalViewBase* self)> onSubmitCallback);

	// Inherited via ModalViewBase
	auto onDraw() -> void override;
	auto setProjectIndex(int index) -> void
	{
		projectIndex_ = index;
	}

	auto projectIndex() -> int
	{
		return projectIndex_;
	}

private:
	EditProjectModel model_{};
	int projectIndex_{ -1 };

public:
	auto setModel(const EditProjectModel& model) -> void
	{
		model_ = model;
	}

	[[nodiscard]] auto model() const -> const EditProjectModel&
	{
		return model_;
	}
};
