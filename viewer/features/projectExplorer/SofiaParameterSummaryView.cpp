#include "SofiaParameterSummaryView.h"

SofiaParameterSummaryView::SofiaParameterSummaryView(ApplicationContext& applicationContext)
	: ModalViewBase{ applicationContext, "Sofia Parameter Summary", ModalType::ok }
{
}
auto SofiaParameterSummaryView::onDraw() -> void
{
	for(auto& [key, value] : params_)
	{
		ImGui::LabelText(key.c_str(), value.c_str());
	}
	unblock();
}
