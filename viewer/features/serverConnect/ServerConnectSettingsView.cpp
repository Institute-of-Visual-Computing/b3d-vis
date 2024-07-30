#define IMGUI_DEFINE_MATH_OPERATORS

#include "ServerConnectSettingsView.h"

#include <array>
#include <format>
#include <vector>

#include "framework/ApplicationContext.h"

#include <IconsLucide.h>

namespace
{
	struct ServerSelection
	{
		std::string name;
	};
	std::vector<ServerSelection> servers = { { "localhost" },
											 { "192.168.0.1" },
											 { "very long server name 192.168.0.1:9999" } };
	std::array<bool, 2> selected = { false, false };
} // namespace

ServerConnectSettingsView::ServerConnectSettingsView(ApplicationContext& appContext, std::string_view name,
													 std::function<void(void)> onSubmitCallback)
	: ModalViewBase(appContext, name, ModalType::okCancel, ImVec2(40 * ImGui::GetFontSize(), 10 * ImGui::GetFontSize()))
{
	setOnSubmit(onSubmitCallback);
}

auto ServerConnectSettingsView::onDraw() -> void
{
	static auto selectedItem = 0;
	ImGuiStyle& style = ImGui::GetStyle();
	float window_visible_x2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
	const auto itemSize = ImVec2{ 200, 200 };
	const auto padding = 10;

	ImGui::BeginChild("##servers", ImVec2{ 0, 300 }, ImGuiChildFlags_Border, ImGuiWindowFlags_AlwaysVerticalScrollbar);
	auto pos = ImGui::GetCursorPos();
	auto s_pos = ImGui::GetCursorPos();


	for (int n = 0; n < servers.size(); n++)
	{
		ImGui::PushID(n);
		ImGui::SetNextItemAllowOverlap();
		ImGui::SetCursorPos(ImVec2(pos.x + padding, pos.y + padding));
		ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 4.0f);
		ImGui::BeginChild("", ImVec2{ 0, 0 },
						  ImGuiChildFlags_Border | ImGuiChildFlags_AlwaysAutoResize | ImGuiChildFlags_AutoResizeX |
							  ImGuiChildFlags_AutoResizeY);

		ImVec2 alignment = ImVec2(0.5f, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, alignment);

		ImGui::PushID(n);
		const auto itemPosition = ImGui::GetCursorPos();
		if (ImGui::Selectable("", selectedItem == n,
							  ImGuiSelectableFlags_DontClosePopups | ImGuiSelectableFlags_AllowOverlap, itemSize))
		{
			selectedItem = n;
		}
		if (ImGui::BeginItemTooltip())
		{
			ImGui::Text(servers[n].name.c_str());
			ImGui::EndTooltip();
		}

		const auto textSize = ImGui::CalcTextSize(servers[n].name.c_str());
		const auto dotsTextSize = ImGui::CalcTextSize("...");
		ImGui::PushFont(applicationContext_->getFontCollection().getBigIconsFont());
		ImGui::SetNextItemAllowOverlap();
		const auto iconSize = ImGui::CalcTextSize(ICON_LC_HARD_DRIVE);

		const auto height = textSize.y + iconSize.y + ImGui::GetStyle().FramePadding.y;

		ImGui::SetCursorPos(itemPosition + ImVec2{ (itemSize.x - iconSize.x) * 0.5f, (itemSize.y - height) * 0.5f });
		ImGui::Text(ICON_LC_HARD_DRIVE);
		ImGui::PopFont();


		if (textSize.x - ImGui::GetStyle().FramePadding.x < itemSize.x)
		{
			ImGui::SetCursorPos(itemPosition +
								ImVec2{ (itemSize.x - textSize.x) * 0.5f,
										(itemSize.y - height) * 0.5f + iconSize.y + ImGui::GetStyle().FramePadding.y });
			ImGui::Text(servers[n].name.c_str());
		}
		else
		{
			auto approximatedLength = servers[n].name.size();
			auto approximatedTextSize = textSize;
			while ((approximatedTextSize.x - ImGui::GetStyle().FramePadding.x) >= itemSize.x)
			{
				approximatedLength /= 2;
				approximatedTextSize =
					ImGui::CalcTextSize(servers[n].name.substr(0, approximatedLength).c_str()) + dotsTextSize;
			}

			const auto text = std::format("{}{}", servers[n].name.substr(0, approximatedLength), "...");
			;

			ImGui::SetCursorPos(itemPosition +
								ImVec2{ (itemSize.x - approximatedTextSize.x) * 0.5f,
										(itemSize.y - height) * 0.5f + iconSize.y + ImGui::GetStyle().FramePadding.y });
			ImGui::Text(text.c_str());
		}


		ImGui::PopID();
		ImGui::PopStyleVar();
		ImGui::EndChild();
		ImGui::PopStyleVar();
		/*	ImGui::PushFont(applicationContext_->getFontCollection().getBigIconsFont());
			ImGui::SetCursorPos(ImVec2(pos.x + padding, pos.y + padding));
			ImGui::Text(ICON_LC_HARD_DRIVE);
			ImGui::PopFont();
			ImGui::SetCursorPos(ImVec2(pos.x + padding, pos.y + padding + 30));
			ImGui::Text(servers[n].name.c_str());*/


		/*	ImGui::SetNextItemAllowOverlap();
			ImGui::SetCursorPos(pos);
			ImGui::Text("Server");
			ImGui::SetNextItemAllowOverlap();
			ImGui::Text(servers[n].name.c_str());
			ImGui::SetCursorPos(pos);*/
		// ImGui::Button("Box", button_sz);
		float last_button_x2 = ImGui::GetItemRectMax().x;
		float next_button_x2 =
			last_button_x2 + style.ItemSpacing.x + itemSize.x; // Expected position if next button was on same line
		if (n + 1 < servers.size() && next_button_x2 < window_visible_x2)
		{
			pos.x = pos.x + padding * 2 + itemSize.x;
		}
		else
		{
			pos.y = pos.y + padding * 2 + itemSize.y;
			pos.x = s_pos.x;
		}
		ImGui::PopID();
	}
	ImGui::EndChild();

	ImGui::Button("Add...");
	ImGui::SameLine();
	ImGui::Button("Edit...");
	ImGui::SameLine();
	ImGui::Button("Remove");


	ImGui::Text("hallo modal!!!");
	if (ImGui::Button("allow submit"))
	{
		unblock();
	}
}
