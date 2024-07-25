#include "MenuBar.h"
#include <imgui.h>
#include <variant>

#include "framework/ApplicationContext.h"

MenuBar::MenuBar(ApplicationContext& applicationContext) : applicationContext_{ &applicationContext }
{
}

auto MenuBar::draw() -> void
{
	ImGui::BeginMainMenuBar();
	for (const auto& [label, groups] : applicationContext_->menuData)
	{
		if (ImGui::BeginMenu(label.c_str()))
		{
			for (const auto& [groupLabel, items] : groups.groups)
			{
				if (groupLabel != "")
				{
					ImGui::SeparatorText(groupLabel.c_str());
				}
				for (auto& item : items)
				{
					std::visit(
						std_help::overloaded{
							[&](const Action& action)
							{
								if (ImGui::MenuItem(item.label.c_str(),
													item.shortcut.has_value() ? item.shortcut.value().data() : nullptr))
								{
									action();
								}
							},
							[&](ApplicationContext::ToggleEntryAction action)
							{
								if (ImGui::MenuItem(item.label.c_str(),
													item.shortcut.has_value() ? item.shortcut.value().data() : nullptr,
													action.value))
								{
									action.onChangeAction(*action.value);
								}
							} },
						item.action);
				}
			}
			ImGui::EndMenu();
		}
	}

	if (!applicationContext_->trayCallbacks.empty())
	{
		ImGui::Spacing();
		ImGui::SameLine(ImGui::GetWindowWidth() - 100);


		for (auto& callback : applicationContext_->trayCallbacks)
		{
			callback();
			ImGui::SameLine();
		}
	}


	ImGui::EndMainMenuBar();
}
