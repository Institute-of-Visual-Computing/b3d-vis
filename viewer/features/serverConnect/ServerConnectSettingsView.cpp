#define IMGUI_DEFINE_MATH_OPERATORS

#include "ServerConnectSettingsView.h"
#include "ServerAddEditView.h"
#include "Style.h"
#include "framework/ApplicationContext.h"

#include <IconsLucide.h>

#include <format>
#include <vector>

#include <imspinner.h>

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
	: ModalViewBase(appContext, name, ModalType::okCancel,
					ImVec2(800 * ImGui::GetFontSize(), 100 * ImGui::GetFontSize()))
{
	setOnSubmit(onSubmitCallback);
	applicationContext_->settings_.load();
	addServerView_ = std::make_unique<ServerAddEditView>(
		appContext, "Add Server", [](ModalViewBase* self)
		{ reinterpret_cast<ServerAddEditView*>(self)->setModel(b3d::tools::project::ServerConnectionDescription{}); },
		[&](ModalViewBase* self)
		{
			const auto model = reinterpret_cast<ServerAddEditView*>(self)->model();
			applicationContext_->settings_.configuredServerSettings_.push_back(model);
			selectedItem_ = static_cast<int>(applicationContext_->settings_.configuredServerSettings_.size() - 1);

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

	removeServerView_ = std::make_unique<ServerRemoveView>(
		appContext, "Remove Server", [&]([[maybe_unused]] ModalViewBase* self) {},
		[&](ModalViewBase* self)
		{
			auto view = reinterpret_cast<ServerRemoveView*>(self);

			applicationContext_->settings_.configuredServerSettings_.erase(
				applicationContext_->settings_.configuredServerSettings_.begin() + view->serverIndex());
		});
}

auto ServerConnectSettingsView::onDraw() -> void
{
	const auto& style = ImGui::GetStyle();
	const auto windowVisibleX2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
	const auto scale = ImGui::GetWindowDpiScale();
	const auto itemSize = ImVec2{ 100 * scale, 100 * scale };
	ImGui::BeginChild("##servers", Vector2{ ImGui::GetContentRegionAvail().x, 400.0f },
					  ImGuiChildFlags_Borders | ImGuiChildFlags_FrameStyle);

	 const auto& servers = applicationContext_->settings_.configuredServerSettings_;
	 for (auto n = 0; n < servers.size(); n++)
	{

		ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 4.0f);
		ImGui::PushID(n);
		const auto itemPosition = ImGui::GetCursorPos();
		if (ui::ToggleButton(selectedItem_ == n, "", itemSize))
		{
			selectedItem_ = n;
			serverClient_ = b3d::tools::project::ServerClient(servers[n]);
		}
		if (ImGui::BeginItemTooltip())
		{
			ImGui::Text(servers[n].name.c_str());
			ImGui::EndTooltip();
		}

		const auto lastButtonX2 = ImGui::GetItemRectMax().x;
		const auto nextButtonX2 = lastButtonX2 + style.ItemSpacing.x + itemSize.x;
		if (n + 1 <= servers.size() && nextButtonX2 < windowVisibleX2)
		{
			ImGui::SameLine();
		}

		const auto textSize = ImGui::CalcTextSize(servers[n].name.c_str());
		const auto dotsTextSize = ImGui::CalcTextSize("...");
		ImGui::PushFont(applicationContext_->getFontCollection().getBigIconsFont());

		const auto iconSize = ImGui::CalcTextSize(ICON_LC_HARD_DRIVE);

		const auto height = textSize.y + iconSize.y + ImGui::GetStyle().FramePadding.y;

		
		const auto nextItemPosition = ImGui::GetCursorPos();
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
		ImGui::SetCursorPos(nextItemPosition);
		
		ImGui::PopID();
		ImGui::PopStyleVar(1);
	}
	ImGui::EndChild();


	const auto addButtonText = "Add";
	const auto addButtonTextSize = ImGui::CalcTextSize(addButtonText, NULL, true);
	const auto addButtonSize = ImGui::CalcItemSize(Vector2{}, addButtonTextSize.x + style.FramePadding.x * 2.0f,
												   addButtonTextSize.y + style.FramePadding.y * 2.0f);

	ImGui::SetNextItemAllowOverlap();
	if (ui::AccentButton(addButtonText, addButtonSize))
	{
		addServerView_->open();
	}
	if (ImGui::BeginItemTooltip())
	{
		ImGui::Text("Add new server connection");
		ImGui::EndTooltip();
	}

	ImGui::BeginDisabled(not isServerSelected());
	{
		ImGui::PushID(selectedItem_);
		const auto emptySpace = ImGui::GetContentRegionAvail().x -
			(ImGui::CalcTextSize(ICON_LC_PENCIL " Edit").x + ImGui::CalcTextSize(ICON_LC_TRASH_2 " Remove").x +
			 style.FramePadding.x * 4.0f + addButtonSize.x + style.ItemSpacing.x);

		ImGui::SameLine(0, emptySpace);

		if (ui::Button(ICON_LC_PENCIL " Edit"))
		{
			editServerView_->open();
		}
		ImGui::SetItemTooltip("Edit Project Name");
		ImGui::SameLine();

		if (ui::Button(ICON_LC_TRASH_2 " Remove"))
		{
			removeServerView_->setServerIndex(selectedItem_);
			removeServerView_->open();
			selectedItem_ = -1;
		}
		ImGui::SetItemTooltip("Removes server connection");
		ImGui::PopID();
	}
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
	removeServerView_->draw();
}

auto ServerConnectSettingsView::testServerStatus() -> void
{
	serverClient_.forceUpdateServerStatusState();
}
