#include "WindowViewBase.h"

#include "IdGenerator.h"

#include <format>
#include <imgui_internal.h>


WindowViewBase::WindowViewBase(const std::string_view name, const WindowFlags flags) : flags_{ flags }
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
		drawContent_ = ImGui::Begin(windowId_.c_str(), hasCloseButton ? &isOpen_ : nullptr, imGuiWindowFlags_);

		if(drawContent_)
		{
			viewportSize_ = ImGui::GetContentRegionAvail();
		}
	}
}

auto WindowViewBase::endDraw() -> void
{
	if (isOpen_)
	{
		ImGui::End();
	}
}
