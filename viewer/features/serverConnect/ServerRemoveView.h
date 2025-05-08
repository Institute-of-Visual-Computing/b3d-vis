#pragma once

#include "framework/ModalViewBase.h"


class ServerRemoveView final : public ModalViewBase
{
public:
	ServerRemoveView(ApplicationContext& appContext, const std::string_view name,
					  std::function<void(ModalViewBase* self)> onOpenCallback,
					  std::function<void(ModalViewBase* self)> onSubmitCallback);

	auto onDraw() -> void override;
	auto setServerIndex(int index) -> void
	{
		serverIndex_ = index;
	}

	auto serverIndex() -> int
	{
		return serverIndex_;
	}

private:
	int serverIndex_{ -1 };
};
