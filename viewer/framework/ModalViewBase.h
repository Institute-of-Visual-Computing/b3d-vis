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
	ModalViewBase(ApplicationContext& applicationContext, const std::string_view name,
				  const ModalType modalType = ModalType::okCancel, const ImVec2& minSize = ImVec2{ 0, 0 });

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

	ApplicationContext* applicationContext_{ nullptr };

private:
	bool blockSubmit_{ true };
	bool isOpenRequested_{ false };

	std::function<void(void)> onSubmitCallback_{};
	ModalType modalType_{};

	std::string id_{};
	ImVec2 minSize_{ 0, 0 };
};
