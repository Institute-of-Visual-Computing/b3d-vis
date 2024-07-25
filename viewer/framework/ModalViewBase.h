#pragma once

#include "WindowViewBase.h"

#include <functional>

enum class ModalType
{
	ok,
	okCancel
};

class ModalViewBase
{
public:
	ModalViewBase(ApplicationContext& appContext, const std::string_view name, const ModalType modalType);

	auto setOnSubmit(std::function<void(void)> callback) -> void
	{
		onSubmitCallback_ = callback;
	}
	auto open() -> void;

	auto draw() -> void;
	auto reset() -> void;

protected:

	virtual auto onDraw() -> void = 0;

	auto block() noexcept -> void
	{
		blockSubmit_ = true;
	}

	auto unblock() noexcept -> void
	{
		blockSubmit_ = false;
	}

	[[nodiscard]] auto isBlocked() const noexcept -> bool
	{
		return blockSubmit_;
	}

	auto submit() -> void;

private:

	bool blockSubmit_{ true };
	bool isOpenRequested_{ false };

	std::function<void(void)> onSubmitCallback_{};
	ModalType modalType_{};

	std::string id_{};
};
