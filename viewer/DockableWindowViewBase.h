#pragma once

#include <cstdint>
#include <string_view>

#include <imgui.h>

#include "WindowViewBase.h"
#include "Dockspace.h"


class DockableWindowViewBase : public WindowViewBase
{
public:
	DockableWindowViewBase(ApplicationContext& appContext, const std::string_view name, Dockspace* dockspace,
						   const WindowFlags flags);

	auto draw() -> void;

private:
	Dockspace* dockspace_{ nullptr };
};
