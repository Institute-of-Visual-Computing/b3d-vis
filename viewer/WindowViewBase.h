#pragma once

#include <cstdint>
#include <imgui.h>
#include <string_view>

#include "Flags.h"



enum class WindowFlagBits : uint16_t
{
	none = 0,
	noUndocking = 1,
	hideTabBar = 2,
	noTitleBar = 4,
	noCollapse = 8,
	noClose = 16,
	noDocking = 32
};

using WindowFlags = Flags<WindowFlagBits>;

inline WindowFlags operator|(const WindowFlagBits& a, const WindowFlagBits& b)
{
	auto flags = WindowFlags{ a };
	flags |= b;
	return flags;
}

class ApplicationContext;

class WindowViewBase
{
public:
	WindowViewBase(ApplicationContext& appContext, const std::string_view name, const WindowFlags flags);

	auto beginDraw() -> void;
	auto endDraw() -> void;

	[[nodiscard]] auto viewportSize() const noexcept -> ImVec2
	{
		return viewportSize_;
	}

	auto open() noexcept -> void
	{
		isOpen_ = true;
	}

	auto close() noexcept -> void
	{
		isOpen_ = false;
	}

protected:
	virtual auto onDraw() -> void = 0;

	ApplicationContext* appContext_{};

	ImGuiWindowClass windowClass_{};

	ImVec2 viewportSize_{};
	std::string windowId_{};
	WindowFlags flags_{ WindowFlagBits::none };
	ImGuiWindowFlags imGuiWindowFlags_{};

	bool isOpen_{ true };
	bool drawContent_{ true };
};
