#include "Dockspace.h"

Dockspace::Dockspace()
{
}

auto Dockspace::begin() -> void
{
	assert(!hasDrawn_);
	hasDrawn_ = true;
	dockspaceId_ = ImGui::DockSpaceOverViewport();
}

auto Dockspace::end() -> void
{
	assert(hasDrawn_);
	hasDrawn_ = false;
}


