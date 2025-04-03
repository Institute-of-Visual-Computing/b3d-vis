#pragma once

#include "framework/ModalViewBase.h"


class DeleteProjectView final : public ModalViewBase
{
public:
	DeleteProjectView(ApplicationContext& appContext, const std::string_view name,
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
	int projectIndex_{ -1 };
};
