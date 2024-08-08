#include "ProjectSelectionView.h"

ProjectSelectionView::ProjectSelectionView(ApplicationContext& applicationContext)
	: ModalViewBase(applicationContext, "Project Selection", ModalType::okCancel)
{
}

auto ProjectSelectionView::onDraw() -> void
{
	ImGui::Text("hallo");
}
