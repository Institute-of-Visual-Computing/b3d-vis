#include "ServerConnectSettingsView.h"

ServerConnectSettingsView::ServerConnectSettingsView(ApplicationContext& appContext, std::string_view name,
													 std::function<void(void)> onSubmitCallback)
	: ModalViewBase(appContext, name, ModalType::okCancel)
{
	setOnSubmit(onSubmitCallback);
}

auto ServerConnectSettingsView::onDraw() -> void
{
	ImGui::Text("hallo modal!!!");
	if (ImGui::Button("allow submit"))
	{
		unblock();
	}
}
