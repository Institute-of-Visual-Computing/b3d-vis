#include "DeleteProjectView.h"
#include "imgui_stdlib.h"

DeleteProjectView::DeleteProjectView(ApplicationContext& appContext, const std::string_view name,
									 std::function<void(ModalViewBase* self)> onOpenCallback,
									 std::function<void(ModalViewBase* self)> onSubmitCallback)
	: ModalViewBase(appContext, name, ModalType::okCancel,
					ImVec2(400 * ImGui::GetFontSize(), 100 * ImGui::GetFontSize()))
{
	setOnOpen(onOpenCallback);
	setOnSubmit(onSubmitCallback);
}

auto DeleteProjectView::onDraw() -> void
{
	ImGui::Text("Do you really want to delete this project?");
	unblock();
}
