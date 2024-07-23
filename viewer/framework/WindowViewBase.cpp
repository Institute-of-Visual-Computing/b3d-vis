#include "WindowViewBase.h"

#include "IdGenerator.h"

#include <format>
#include <imgui_internal.h>


WindowViewBase::WindowViewBase(ApplicationContext& appContext, const std::string_view name, const WindowFlags flags)
	: flags_{ flags }, applicationContext_{ &appContext }
{

	windowId_ = std::format("{}##{}", name, IdGenerator::next());


	if ((flags_ & WindowFlagBits::noTitleBar) == WindowFlagBits::noTitleBar)
	{
		imGuiWindowFlags_ |= ImGuiWindowFlags_NoTitleBar;
	}
	if ((flags_ & WindowFlagBits::noCollapse) == WindowFlagBits::noCollapse)
	{
		imGuiWindowFlags_ |= ImGuiWindowFlags_NoCollapse;
	}
	if ((flags_ & WindowFlagBits::noDocking) == WindowFlagBits::noDocking)
	{
		windowClass_.DockNodeFlagsOverrideSet =
			ImGuiDockNodeFlags_NoDockingSplitOther | ImGuiDockNodeFlags_NoDocking;
	}
}

auto WindowViewBase::beginDraw() -> void
{
	ImGui::SetNextWindowClass(&windowClass_);
	if (isOpen_)
	{
		auto hasCloseButton = (flags_ & WindowFlagBits::noClose) != WindowFlagBits::noClose;
		drawContent_ = ImGui::Begin(windowId_.c_str(), hasCloseButton ? &isOpen_ : nullptr,
									imGuiWindowFlags_);

		if(drawContent_)
		{
			const auto viewportSize = ImGui::GetContentRegionAvail();
			if ((viewportSize.x != viewportSize_.x) || (viewportSize.y != viewportSize_.y))
			{
				needResize_ = true;
			}
			else
			{
				needResize_ = false;
			}
			viewportSize_ = viewportSize;
		}
	}
}

auto WindowViewBase::endDraw() -> void
{
	if (isOpen_)
	{
		ImGui::End();
	}
	if (needResize_)
	{
		onResize();
	}
}

auto WindowViewBase::isVisible() const -> bool
{
	return drawContent_ && (viewportSize_.x > 0 && viewportSize_.y > 0);
}
