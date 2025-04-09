#include "DockableWindowViewBase.h"

#include <format>


DockableWindowViewBase::DockableWindowViewBase(ApplicationContext& appContext, const std::string_view name,
											   Dockspace* dockspace, const WindowFlags flags)
	: WindowViewBase{ appContext, name, flags }, dockspace_{ dockspace }
{
	windowClass_.ClassId = dockspace_->id();
	windowClass_.DockingAllowUnclassed = true;

	if ((flags_ & WindowFlagBits::hideTabBar) == WindowFlagBits::hideTabBar)
	{
		windowClass_.DockNodeFlagsOverrideSet |= ImGuiDockNodeFlags_AutoHideTabBar;
	}

	if ((flags_ & WindowFlagBits::noUndocking) == WindowFlagBits::noUndocking)
	{
		//windowClass_.DockNodeFlagsOverrideSet |= ImGuiDockNodeFlags_NoUndocking;
	}
}

auto DockableWindowViewBase::draw() -> void
{
	assert(dockspace_->hasDrawn());
	ImGui::SetNextWindowDockID(dockspace_->id(), ImGuiCond_FirstUseEver);
	beginDraw();
	if (isOpen_ and drawContent_)
	{
		onDraw();
	}
	endDraw();
}
