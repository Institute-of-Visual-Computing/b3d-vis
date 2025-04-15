#pragma once

#include "WindowViewBase.h"

#include <functional>
#include <string>

enum class ModalType
{
	ok,
	okCancel
};

class ModalViewBase
{
public:
	virtual ~ModalViewBase() = default;
	ModalViewBase(ApplicationContext& applicationContext, const std::string_view name,
				  const ModalType modalType = ModalType::okCancel, const ImVec2& minSize = ImVec2{ 0, 0 });

	auto setOnSubmit(const std::function<void(ModalViewBase* self)>& callback) -> void
	{
		onSubmitCallback_ = callback;
	}

	auto setOnOpen(const std::function<void(ModalViewBase* self)>& callback) -> void
	{
		onOpenCallback_ = callback;
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

	std::function<void(ModalViewBase* self)> onSubmitCallback_{};
	std::function<void(ModalViewBase* self)> onOpenCallback_{};
	ModalType modalType_{};

	std::string id_{};
	std::string name_{};
	ImVec2 minSize_{ 0, 0 };
};
