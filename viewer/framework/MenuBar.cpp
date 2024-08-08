#include "MenuBar.h"
#include <imgui.h>
#include <variant>

#include "framework/ApplicationContext.h"

MenuBar::MenuBar(ApplicationContext& applicationContext) : applicationContext_{ &applicationContext }
{
}

auto MenuBar::draw() const -> void
{
	ImGui::BeginMainMenuBar();
	for (const auto& [label, groups] : applicationContext_->menuData_)
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
							[&](const ApplicationContext::ToggleEntryAction& action)
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

	if (!applicationContext_->trayCallbacks_.empty())
	{
		ImGui::Spacing();
		ImGui::SameLine(ImGui::GetWindowWidth() - 100);


		for (auto& callback : applicationContext_->trayCallbacks_)
		{
			callback();
			ImGui::SameLine();
		}
	}


	ImGui::EndMainMenuBar();
}
