#include "ServerRemoveView.h"
#include "imgui_stdlib.h"

ServerRemoveView::ServerRemoveView(ApplicationContext& appContext, const std::string_view name,
									 std::function<void(ModalViewBase* self)> onOpenCallback,
									 std::function<void(ModalViewBase* self)> onSubmitCallback)
	: ModalViewBase(appContext, name, ModalType::okCancel,
					ImVec2(400 * ImGui::GetFontSize(), 100 * ImGui::GetFontSize()))
{
	setOnOpen(onOpenCallback);
	setOnSubmit(onSubmitCallback);
}

auto ServerRemoveView::onDraw() -> void
{
	ImGui::Text("Do you really want to remove the server connection?");
	unblock();
}
