#define IMGUI_DEFINE_MATH_OPERATORS

#include "ServerConnectSettingsView.h"
#include "ServerAddEditView.h"
#include "framework/ApplicationContext.h"

#include <IconsLucide.h>

#include <format>
#include <vector>

#include "imspinner.h"

namespace
{
	/*struct ServerSelection
	{
		std::string name;
	};
	std::vector<ServerSelection> servers = { { "localhost" },
											 { "192.168.0.1" },
											 { "very long server name 192.168.0.1:9999" } };
	*/
	auto counter = 0;
	auto startTestConnection = false;
} // namespace

ServerConnectSettingsView::ServerConnectSettingsView(ApplicationContext& appContext, const std::string_view name,
													 const std::function<void(ModalViewBase*)>& onSubmitCallback)
	: ModalViewBase(appContext, name, ModalType::okCancel, ImVec2(40 * ImGui::GetFontSize(), 10 * ImGui::GetFontSize()))
{
	setOnSubmit(onSubmitCallback);
	applicationContext_->settings_.load();
	addServerView_ = std::make_unique<ServerAddEditView>(
		appContext, "Add Server",
		[](ModalViewBase* self)
		{ reinterpret_cast<ServerAddEditView*>(self)->setModel(b3d::tools::project::ServerConnectionDescription{}); },
		[&](ModalViewBase* self)
		{
			const auto model = reinterpret_cast<ServerAddEditView*>(self)->model();
			applicationContext_->settings_.configuredServerSettings_.push_back(model);
			selectedItem_ = applicationContext_->settings_.configuredServerSettings_.size() - 1;
			
			serverClient_ = b3d::tools::project::ServerClient(model);
			applicationContext_->settings_.save();
		});

	editServerView_ = std::make_unique<ServerAddEditView>(
		appContext, "Edit Server",
		[&](ModalViewBase* self)
		{
			    reinterpret_cast<ServerAddEditView*>(self)->setModel(
				applicationContext_->settings_.configuredServerSettings_[selectedItem_]);
		},
		[&](ModalViewBase* self)
		{
			const auto model = reinterpret_cast<ServerAddEditView*>(self)->model();
			applicationContext_->settings_.configuredServerSettings_[selectedItem_] = model;
			serverClient_ = b3d::tools::project::ServerClient(model);
			applicationContext_->settings_.save();
		});
}

auto ServerConnectSettingsView::onDraw() -> void
{
	const auto& style = ImGui::GetStyle();
	const auto windowVisibleX2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
	constexpr auto itemSize = ImVec2{ 200, 200 };

	ImGui::BeginChild("##servers", ImVec2{ 0, 300 }, ImGuiChildFlags_Border, ImGuiWindowFlags_AlwaysVerticalScrollbar);
	auto pos = ImGui::GetCursorPos();
	const auto widgetStartPosition = ImGui::GetCursorPos();

	const auto& servers = applicationContext_->settings_.configuredServerSettings_;
	for (auto n = 0; n < servers.size(); n++)
	{
		constexpr auto padding = 10;
		ImGui::PushID(n);
		ImGui::SetNextItemAllowOverlap();
		ImGui::SetCursorPos(ImVec2(pos.x + padding, pos.y + padding));
		ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 4.0f);
		ImGui::BeginChild("", ImVec2{ 0, 0 },
						  ImGuiChildFlags_Border | ImGuiChildFlags_AlwaysAutoResize | ImGuiChildFlags_AutoResizeX |
							  ImGuiChildFlags_AutoResizeY);

		auto alignment = ImVec2(0.5f, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, alignment);

		ImGui::PushID(n);
		const auto itemPosition = ImGui::GetCursorPos();
		if (ImGui::Selectable("", selectedItem_ == n,
							  ImGuiSelectableFlags_DontClosePopups | ImGuiSelectableFlags_AllowOverlap, itemSize))
		{
			selectedItem_ = n;
			serverClient_ = b3d::tools::project::ServerClient(servers[n]);
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
		const auto lastButtonX2 = ImGui::GetItemRectMax().x;
		const auto nextButtonX2 = lastButtonX2 + style.ItemSpacing.x + itemSize.x;
		if (n + 1 < servers.size() && nextButtonX2 < windowVisibleX2)
		{
			pos.x = pos.x + padding * 2 + itemSize.x;
		}
		else
		{
			pos.y = pos.y + padding * 2 + itemSize.y;
			pos.x = widgetStartPosition.x;
		}
		ImGui::PopID();
	}
	ImGui::EndChild();

	if (ImGui::Button("Add..."))
	{
		addServerView_->open();
	}
	ImGui::SameLine();

	ImGui::BeginDisabled(!isServerSelected());
	if (ImGui::Button("Edit..."))
	{
		editServerView_->open();
	}

	ImGui::SameLine();
	if (ImGui::Button("Remove"))
	{
		applicationContext_->settings_.configuredServerSettings_.erase(
			applicationContext_->settings_.configuredServerSettings_.begin() + selectedItem_);
		selectedItem_ -= 1;
	}
	ImGui::SameLine(ImGui::GetContentRegionAvail().x - 225);

	ImGui::BeginGroup();

	if (ImGui::Button("Set"))
	{
		applicationContext_->serverClient_.setNewConnectionInfo(serverClient_.getConnectionInfo());
	}
	ImGui::SameLine();
	if (ImGui::Button("Test Connection"))
	{
		testServerStatus();
	}

	if (isServerSelected())
	{
		ImGui::SameLine();
		switch (serverClient_.getLastServerStatusState().health)
		{

		case b3d::tools::project::ServerHealthState::ok:
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 0.1f, 0.7f, 0.1f, 1.0f });
			ImGui::Text(ICON_LC_CIRCLE_CHECK);
			ImGui::PopStyleColor();
			break;
		case b3d::tools::project::ServerHealthState::unreachable:
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 0.7f, 0.1f, 0.1f, 1.0f });
			ImGui::Text(ICON_LC_SERVER_CRASH);
			ImGui::PopStyleColor();
			break;
		case b3d::tools::project::ServerHealthState::unknown:
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 0.7f, 0.7f, 0.1f, 1.0f });
			ImGui::Text(ICON_LC_TRIANGLE_ALERT);
			ImGui::PopStyleColor();
			break;
		case b3d::tools::project::ServerHealthState::testing:
			ImSpinner::SpinnerRotateSegments("server_test_spinner", 8, 2.0f);
			break;
		}
	}
	

	ImGui::EndGroup();
	ImGui::EndDisabled();


	if (isServerSelected())
	{
		unblock();
	}
	else
	{
		block();
	}

	addServerView_->draw();
	editServerView_->draw();
}

auto ServerConnectSettingsView::testServerStatus() -> void
{
	serverClient_.forceUpdateServerStatusState();
}
