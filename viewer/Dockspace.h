#pragma once

#include <imgui.h>

class Dockspace
{
public:
	Dockspace();

	auto begin() -> void;
	auto end() -> void;


	[[nodiscard]] auto hasDrawn() const -> bool
	{
		return hasDrawn_;
	}

	[[nodiscard]] auto id() const -> ImGuiID
	{
		return dockspaceId_;
	}

private:
	ImGuiID dockspaceId_{};
	bool hasDrawn_{ false };
};
