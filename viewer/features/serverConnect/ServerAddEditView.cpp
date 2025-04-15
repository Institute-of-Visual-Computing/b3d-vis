#include "ServerAddEditView.h"

#include "framework/ApplicationContext.h"

#include "Style.h"

ServerAddEditView::ServerAddEditView(ApplicationContext& applicationContext, const std::string_view name,
									 std::function<void(ModalViewBase*)> onOpenCallback,
									 std::function<void(ModalViewBase*)> onSubmitCallback)
	: ModalViewBase(applicationContext, name, ModalType::okCancel,
					ImVec2(20 * ImGui::GetFontSize(), 10 * ImGui::GetFontSize()))
{
	setOnOpen(onOpenCallback);
	setOnSubmit(onSubmitCallback);
}

auto ServerAddEditView::onDraw() -> void
{
	ui::HeadedInputText("Name:", "##name", &model_.name);
	ui::HeadedInputText("IP Address:", "##ip_address", &model_.ipHost);
	ui::HeadedInputText("Port:", "##port", &model_.port);

	unblock();
}
