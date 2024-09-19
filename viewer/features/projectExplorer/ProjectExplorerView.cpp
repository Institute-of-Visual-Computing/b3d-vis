#define IMGUI_DEFINE_MATH_OPERATORS
#include "Project.h"

#include "framework/ApplicationContext.h"

#include "ProjectExplorerView.h"

#include "DebugDrawList.h"
#include "IconsLucide.h"
#include "IdGenerator.h"
#include "ImGuiExtension.h"
#include "imspinner.h"

namespace
{

	struct ProjectItem
	{
		std::string name;
	};

	std::vector<ProjectItem> model;

	void populateTestModelData()
	{
		model.push_back({ ProjectItem{ "b3d_test_1" } });
		model.push_back({ ProjectItem{ "b3d_test_1_very_long_name___________________" } });
		model.push_back({ ProjectItem{ "b3d_test_2_very_long_name___________________" } });
		model.push_back({ ProjectItem{ "b3d_test_3_very_long_name___________________" } });
		model.push_back({ ProjectItem{ "b3d_test_4_very_long_name___________________" } });
		model.push_back({ ProjectItem{ "b3d_test_5_very_long_name___________________" } });
		model.push_back({ ProjectItem{ "b3d_test_6_very_long_name___________________" } });
		model.push_back({ ProjectItem{ "b3d_test_7_very_long_name___________________" } });
	}

} // namespace

ProjectExplorerView::ProjectExplorerView(
	ApplicationContext& appContext, Dockspace* dockspace, std::function<void()> showSelectionModal,
	std::function<void()> showNvdbSelectionModal,
	std::function<std::shared_future<void>(const std::string& fileUUID)> loadAndShowFunction,
	std::function<std::shared_future<void>()> refreshProjectsFunction)
	: DockableWindowViewBase(appContext, "Project Explorer", dockspace, WindowFlagBits::none),
	  showSelectionModal_(std::move(showSelectionModal)), showNvdbSelectionModal_(std::move(showNvdbSelectionModal)),
	  loadAndShowFunction_(std::move(loadAndShowFunction)), refreshProjectsFunction_(std::move(refreshProjectsFunction))
{
	populateTestModelData();
}

ProjectExplorerView::~ProjectExplorerView() = default;

auto ProjectExplorerView::setModel(Model model) -> void
{
	model_ = std::move(model);
}

auto drawSelectableItemGridPanel(
	const char* panelId, int& selectedItemIndex, const int items,
	const std::function<const char*(const int index)>& name, const char* icon, ImFont* iconFont,
	const std::function<void(const int index)>& popup = [](const int) {}, const ImVec2 itemSize = { 100, 100 },
	const ImVec2 panelSize = { 0, 300 })
{
	const auto& style = ImGui::GetStyle();
	const auto windowVisibleX2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
	ImGui::PushID(panelId);
	ImGui::BeginChild("", panelSize, ImGuiChildFlags_Border, ImGuiWindowFlags_AlwaysVerticalScrollbar);
	auto pos = ImGui::GetCursorPos();
	const auto widgetStartPosition = ImGui::GetCursorPos();


	for (auto n = 0; n < items; n++)
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
		if (ImGui::Selectable("", selectedItemIndex == n,
							  ImGuiSelectableFlags_DontClosePopups | ImGuiSelectableFlags_AllowOverlap, itemSize))
		{
			selectedItemIndex = n;
		}
		if (ImGui::BeginItemTooltip())
		{
			popup(n);
			ImGui::EndTooltip();
		}

		const auto textSize = ImGui::CalcTextSize(name(n));
		const auto dotsTextSize = ImGui::CalcTextSize("...");
		ImGui::PushFont(iconFont);
		ImGui::SetNextItemAllowOverlap();
		const auto iconSize = ImGui::CalcTextSize(icon);

		const auto height = textSize.y + iconSize.y + ImGui::GetStyle().FramePadding.y;

		ImGui::SetCursorPos(itemPosition + ImVec2{ (itemSize.x - iconSize.x) * 0.5f, (itemSize.y - height) * 0.5f });
		ImGui::Text(icon);
		ImGui::PopFont();


		if (textSize.x - ImGui::GetStyle().FramePadding.x < itemSize.x)
		{
			ImGui::SetCursorPos(itemPosition +
								ImVec2{ (itemSize.x - textSize.x) * 0.5f,
										(itemSize.y - height) * 0.5f + iconSize.y + ImGui::GetStyle().FramePadding.y });
			ImGui::Text(name(n));
		}
		else
		{
			const auto nameText = std::string{ name(n) };
			auto approximatedLength = nameText.size();
			auto approximatedTextSize = textSize;
			while ((approximatedTextSize.x - ImGui::GetStyle().FramePadding.x) >= itemSize.x)
			{
				approximatedLength /= 2;
				approximatedTextSize =
					ImGui::CalcTextSize(nameText.substr(0, approximatedLength).c_str()) + dotsTextSize;
			}

			const auto text = std::format("{}{}", nameText.substr(0, approximatedLength), "...");
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
		if (n + 1 < items && nextButtonX2 < windowVisibleX2)
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
	ImGui::PopID();
}


auto ProjectExplorerView::onDraw() -> void
{
	const auto availableWidth = ImGui::GetContentRegionAvail().x;


	const auto isConnectedToAnyServer =
		applicationContext_->serverClient_.getLastServerStatusState() == b3d::tools::project::ServerStatusState::ok;
	const auto serverNameText = std::format(
		"Server: {}",
		isConnectedToAnyServer ? applicationContext_->serverClient_.getConnectionInfo().name : "Disconnected!");
	const auto textSize = ImGui::CalcTextSize(serverNameText.c_str());


	const auto refreshedPressed = ImGui::Button(ICON_LC_REFRESH_CW);
	const auto buttonWidth = ImGui::CalcTextSize(ICON_LC_REFRESH_CW).x;
	const auto middleSpace = availableWidth - textSize.x - 2 * buttonWidth;

	ImGui::SetItemTooltip("Refresh");
	ImGui::SameLine(middleSpace);
	ImGui::Text(serverNameText.c_str());
	ImGui::SameLine();
	const auto switchServerPressed = ImGui::Button(ICON_LC_ARROW_RIGHT_LEFT);
	ImGui::SetItemTooltip("Switch Server");

	if (applicationContext_->isDevelopmentModeEnabled)
	{


		ImGui::BeginUnderDevelopmentScope();
		if (ImGui::Button("Load .nvdb manually"))
		{
			showNvdbSelectionModal_();
		}
		ImGui::EndUnderDevelopmentScope();
	}
	if (isConnectedToAnyServer and not projectAvailable())
	{
		static auto timer = 0.0f;

		if (timer <= 0.0f)
		{
			refreshProjectsFuture_ = refreshProjectsFunction_();
			timer = 5.0f;
		}
		timer -= ImGui::GetIO().DeltaTime;
	}

	if (refreshedPressed)
	{
		refreshProjectsFuture_ = refreshProjectsFunction_();
	}
	if (refreshProjectsFuture_.valid() and
		(isConnectedToAnyServer or
		 applicationContext_->serverClient_.getLastServerStatusState() ==
			 b3d::tools::project::ServerStatusState::testing))
	{
		if (refreshProjectsFuture_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
		{
			ImSpinner::SpinnerRotateSegments("project_loading_spinner", ImGui::GetFontSize() * 0.5f, 2.0f);
			ImGui::Text("Loading");
		}
		else
		{
			refreshProjectsFuture_ = {};
		}
	}

	if (model_.projects)
	{
		drawSelectableItemGridPanel(
			"projects", selectedProjectItemIndex_, model_.projects->size(),
			[&](const int index) { return model_.projects->at(index).projectName.c_str(); }, ICON_LC_BOX,
			applicationContext_->getFontCollection().getBigIconsFont(),
			[&](const int index)
			{
				ImGui::Text(model_.projects->at(index).projectName.c_str());
				ImGui::LabelText(model_.projects->at(index).fitsOriginFileName.c_str(), "Source File");
				for (auto i = 0; i < model_.projects->at(index).fitsOriginProperties.axisTypes.size(); i++)
				{
					ImGui::LabelText(std::format("Axis {}", i).c_str(),
									 model_.projects->at(index).fitsOriginProperties.axisTypes[i].c_str());
				}
			},
			ImVec2{ 100, 100 });

		if (selectedProjectItemIndex_ >= 0)
		{
			const auto& project = model_.projects->at(selectedProjectItemIndex_);
			ImGui::Text(project.projectName.c_str());
			if (ImGui::CollapsingHeader(project.fitsOriginFileName.c_str()))
			{
				ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
				// window_flags |= ImGuiWindowFlags_MenuBar;
				ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
				ImGui::BeginChild("Properties", ImVec2(0, 260), ImGuiChildFlags_Border, window_flags);

				// Common Properties
				// Size
				// Properties per Axis
				for (auto i = 0; i < project.fitsOriginProperties.axisTypes.size(); i++)
				{
					ImGui::LabelText(std::format("Axis {}", i).c_str(),
									 project.fitsOriginProperties.axisTypes[i].c_str());
				}

				ImGui::EndChild();
				ImGui::PopStyleVar();
			}


			// TODO: list, item "name", "id", "request status", Button "Jump TO" and highlight in volume view on
			// hover

			static auto selectedRequest = -1;

			if (ImGui::TreeNode("Requests"))
			{
				for (auto i = 0; i < project.requests.size(); i++)
				{
					const auto& request = project.requests[i];
					ImGui::SetNextItemAllowOverlap();
					if (ImGui::Selectable(request.uuid.c_str(), selectedRequest == i))
					{
						selectedRequest = i;
					}
					if (ImGui::IsItemHovered())
					{
						ImGui::SameLine();
						ImGui::Text("Hovered");
						const auto box = owl::common::box3f{ { static_cast<float>(request.subRegion.lower.x),
															   static_cast<float>(request.subRegion.lower.y),
															   static_cast<float>(request.subRegion.lower.z) },

															 { static_cast<float>(request.subRegion.upper.x),
															   static_cast<float>(request.subRegion.upper.y),
															   static_cast<float>(request.subRegion.upper.z) } };

						constexpr auto blinkFrequency = 10.0f;
						const auto blinkIntensity =
							0.5f + 0.5f * glm::sin(ImGui::GetCurrentContext()->HoveredIdTimer * blinkFrequency);


						applicationContext_->getDrawList()->drawBox(
							volumeTransform_.p / 2, volumeTransform_.p, box.size(),
							{ 1.0, 0.0, 0.0,
							  1.0f -
								  blinkIntensity * blinkIntensity * blinkIntensity * blinkIntensity * blinkIntensity },
							volumeTransform_.l);
					}
					ImGui::SameLine();
					if (ImGui::SmallButton("Go To"))
					{
					}
				}
				ImGui::TreePop();
			}


			constexpr auto tableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
				ImGuiTableFlags_SizingFixedFit;

			// ID, Success, Cached, Load & Show
			if (ImGui::BeginTable("RequestTable", 4, tableFlags))
			{
				ImGui::TableSetupColumn("ID");
				ImGui::TableSetupColumn("Success");
				ImGui::TableSetupColumn("Cached");
				ImGui::TableSetupColumn("Load & Show");
				ImGui::TableHeadersRow();

				auto blockLoadGet = false;
				if (loadAndShowFileFuture_.valid())
				{
					blockLoadGet = true;
					if (loadAndShowFileFuture_.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
					{
						loadAndShowFileFuture_.get();
						blockLoadGet = false;
					}
				}


				for (const auto& request : project.requests)
				{
					const auto reqSucc = request.result.wasSuccess();

					ImGui::TableNextRow();
					ImGui::PushID(request.uuid.c_str());
					ImGui::TableNextColumn();
					ImGui::Text(request.uuid.c_str());

					ImGui::TableNextColumn();
					ImGui::Text(reqSucc ? "true" : "false");

					ImGui::TableNextColumn();
					if (reqSucc)
					{
						// Replace with icon?
						ImGui::Text("false");
					}
					else
					{
						ImGui::Text("N/A");
					}

					ImGui::TableNextColumn();
					if (!reqSucc || blockLoadGet)
					{
						ImGui::BeginDisabled(true);
					}

					if (ImGui::Button("Load & Show"))
					{
						// Load & Show
						loadAndShowFileFuture_ = loadAndShowFunction_(request.result.nanoResult.resultFile);
					}
					if (!reqSucc || blockLoadGet)
					{
						ImGui::EndDisabled();
					}
					ImGui::PopID();
				}
				ImGui::EndTable();
			}
		}
	}
}

auto ProjectExplorerView::projectAvailable() const -> bool
{
	return model_.projects != nullptr;
}
