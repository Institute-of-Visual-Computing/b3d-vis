#pragma once

#include "WindowViewBase.h"

#include <functional>

enum class ModalType
{
	ok,
	okCancel
};

class ModalViewBase : public WindowViewBase
{
public:
	ModalViewBase(const std::string_view name, const ModalType modalType);

	auto setOnSubmit(std::function<void(void)> callback) -> void
	{
		onSubmitCallback_ = callback;
	}

	auto draw() -> void;
	auto reset() -> void;

protected:
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
	auto requestClose() -> void
	{
		closeRequested_ = true;
	}

	bool blockSubmit_{ true };
	bool closeRequested_{ false };

	std::function<void(void)> onSubmitCallback_{};
	ModalType modalType_{};
};
