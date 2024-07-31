#include "ServerAddEditView.h"

#include "framework/ApplicationContext.h"

ServerAddEditView::ServerAddEditView(ApplicationContext& applicationContext, const std::string_view name,
									 std::function<void(void)> onSubmitCallback)
	: ModalViewBase(applicationContext, name, ModalType::okCancel,
					ImVec2(40 * ImGui::GetFontSize(), 10 * ImGui::GetFontSize())),
	  UpdatableComponentBase(applicationContext)
{
}

auto ServerAddEditView::onDraw() -> void
{
	static char adr_buffer[256];
	ImGui::InputText("address", adr_buffer, 256);
}

auto ServerAddEditView::update() -> void
{
	draw();
}
