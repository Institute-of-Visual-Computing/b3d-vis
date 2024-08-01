#include "ServerAddEditView.h"

#include "framework/ApplicationContext.h"

#include "imgui_stdlib.h"

ServerAddEditView::ServerAddEditView(ApplicationContext& applicationContext, const std::string_view name,
									 std::function<void(ModalViewBase*)> onOpenCallback,
									 std::function<void(ModalViewBase*)> onSubmitCallback)
	: ModalViewBase(applicationContext, name, ModalType::okCancel,
					ImVec2(40 * ImGui::GetFontSize(), 10 * ImGui::GetFontSize()))
{
	setOnOpen(onOpenCallback);
	setOnSubmit(onSubmitCallback);
}

auto ServerAddEditView::onDraw() -> void
{
	ImGui::InputText("Name", &model_.name);
	ImGui::InputText("IP Address", &model_.ip);
	ImGui::InputText("Port", &model_.port);

	unblock();
}
