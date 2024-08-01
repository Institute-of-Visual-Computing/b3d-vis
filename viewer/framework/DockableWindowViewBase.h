#pragma once

#include <string_view>

#include "Dockspace.h"
#include "WindowViewBase.h"


class DockableWindowViewBase : public WindowViewBase
{
public:
	DockableWindowViewBase(ApplicationContext& appContext, std::string_view name, Dockspace* dockspace,
						   WindowFlags flags);

	auto draw() -> void;

private:
	Dockspace* dockspace_{ nullptr };
};
