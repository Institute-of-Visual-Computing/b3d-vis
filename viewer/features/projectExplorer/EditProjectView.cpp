#include "EditProjectView.h"
#include "Style.h"

EditProjectView::EditProjectView(ApplicationContext& appContext, const std::string_view name,
								 std::function<void(ModalViewBase* self)> onOpenCallback,
								 std::function<void(ModalViewBase* self)> onSubmitCallback)
	: ModalViewBase(appContext, name, ModalType::okCancel,
					ImVec2(400 * ImGui::GetFontSize(), 100 * ImGui::GetFontSize()))
{
	setOnOpen(onOpenCallback);
	setOnSubmit(onSubmitCallback);
}

auto EditProjectView::onDraw() -> void
{
	ImGui::PushItemWidth(-1.0f);
	ui::HeadedInputText("Name:", "##Project Name", &model_.projectName);
	ImGui::PopItemWidth();
	if (not model_.projectName.empty())
	{
		unblock();
	}
}
