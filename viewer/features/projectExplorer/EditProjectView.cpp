#include "EditProjectView.h"
#include "imgui_stdlib.h"

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
	ImGui::InputText("Project Name", &model_.projectName);

	if (not model_.projectName.empty())
	{
		unblock();
	}
}
