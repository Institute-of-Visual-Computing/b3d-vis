#include "Dockspace.h"

Dockspace::Dockspace()
{
	dockspaceId_ = ImGui::GetID("mainDock");
}

auto Dockspace::begin() -> void
{
	assert(!hasDrawn_);
	hasDrawn_ = true;
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->WorkPos);
	ImGui::SetNextWindowSize(viewport->WorkSize);
	ImGui::Begin("Editor", 0,
				 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus);
	ImGui::End();
}

auto Dockspace::end() -> void
{
	assert(hasDrawn_);
	hasDrawn_ = false;
}


