#define IMGUI_DEFINE_MATH_OPERATORS
#include "Project.h"

#include "framework/ApplicationContext.h"

#include "ProjectExplorerView.h"

#include "IconsLucide.h"
#include "ImGuiExtension.h"

ProjectExplorerView::ProjectExplorerView(
	ApplicationContext& appContext, Dockspace* dockspace, std::function<void()> showSelectionModal,
	std::function<void()> showNvdbSelectionModal,
	std::function<std::shared_future<void>(const std::string& fileUUID)> loadAndShowFunction)
	: DockableWindowViewBase(appContext, "Project Explorer", dockspace, WindowFlagBits::none),
	  showSelectionModal_(std::move(showSelectionModal)), showNvdbSelectionModal_(std::move(showNvdbSelectionModal)),
	  loadAndShowFunction_(std::move(loadAndShowFunction))
{
}

ProjectExplorerView::~ProjectExplorerView() = default;

auto ProjectExplorerView::setModel(Model model) -> void
{
	model_ = std::move(model);
}

auto ProjectExplorerView::onDraw() -> void
{
	const auto availableWidth = ImGui::GetContentRegionAvail().x;

	if (applicationContext_->serverClient_.getLastServerStatusState() != b3d::tools::project::ServerStatusState::ok)
	{
		ImGui::Text("Server not connected!");
		return;
	}

	const auto serverNameText = std::format("Server: {}", applicationContext_->serverClient_.getConnectionInfo().name);
	const auto textSize = ImGui::CalcTextSize(serverNameText.c_str());


	ImGui::Button(ICON_LC_REFRESH_CW);
	const auto buttonWidth = ImGui::CalcTextSize(ICON_LC_REFRESH_CW).x;
	const auto middleSpace = availableWidth - textSize.x - 2 * buttonWidth;

	ImGui::SetItemTooltip("Refresh");
	ImGui::SameLine(middleSpace);
	ImGui::Text(serverNameText.c_str());
	ImGui::SameLine();
	const auto switchServerPressed = ImGui::Button(ICON_LC_ARROW_RIGHT_LEFT);
	ImGui::SetItemTooltip("Switch Server");


	ImGui::BeginUnderDevelopmentScope();
	if (ImGui::Button("Load .nvdb manually"))
	{
		showNvdbSelectionModal_();
	}
	ImGui::EndUnderDevelopmentScope();
	if (!projectAvailable())
	{
		ImGui::Text("No project selected.");
		if (ImGui::Button("Select Project"))
		{
			showSelectionModal_();
		}
		return;
	}
	ImGui::Text(model_.project->projectName.c_str());
	if (ImGui::CollapsingHeader(model_.project->fitsOriginFileName.c_str()))
	{
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
		// window_flags |= ImGuiWindowFlags_MenuBar;
		ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
		ImGui::BeginChild("Properties", ImVec2(0, 260), ImGuiChildFlags_Border, window_flags);

		// Common Properties
		// Size
		// Properties per Axis
		for (auto i = 0; i < model_.project->fitsOriginProperties.axisTypes.size(); i++)
		{
			ImGui::LabelText(std::format("Axis {}", i).c_str(),
							 model_.project->fitsOriginProperties.axisTypes[i].c_str());
		}

		ImGui::EndChild();
		ImGui::PopStyleVar();
	}

	constexpr auto tableFlags =
		ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit;

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

		for (const auto& request : model_.project->requests)
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

auto ProjectExplorerView::projectAvailable() const -> bool
{
	return model_.project != nullptr;
}
