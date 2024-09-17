#include <chrono>
#include <format>

#include <imspinner.h>

#include "framework/ApplicationContext.h"

#include "ProjectSelectionView.h"

ProjectSelectionView::ProjectSelectionView(ApplicationContext& applicationContext,
										   std::function<std::shared_future<void>()> refreshProjectsFunction,
										   std::function<void(ModalViewBase* self)> onOpenCallback,
										   std::function<void(ModalViewBase* self)> onSubmitCallback)
	: ModalViewBase(applicationContext, "Project Selection", ModalType::okCancel,
					ImVec2(40 * ImGui::GetFontSize(), 10 * ImGui::GetFontSize())),
	  refreshProjectsFunction_(std::move(refreshProjectsFunction))
{
	setOnOpen(onOpenCallback);
	setOnSubmit(onSubmitCallback);
}

auto ProjectSelectionView::onDraw() -> void
{

	const auto& style = ImGui::GetStyle();
	const auto windowVisibleX2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
	auto la = ImGui::GetContentRegionAvail();
	ImGui::Text(availableProjectString_.c_str());

	ImGui::SameLine();
	const auto startFrameRequestOngoing = requestOngoing();
	if (startFrameRequestOngoing)
	{
		ImGui::BeginDisabled(true);
	}

	if (ImGui::Button("Reload"))
	{
		refreshProjectsFuture_ = refreshProjectsFunction_();
	}

	if (startFrameRequestOngoing)
	{
		ImGui::EndDisabled();
		if (refreshProjectsFuture_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
		{

			ImGui::SameLine();
			ImSpinner::SpinnerRotateSegments("project_loading_spinner", ImGui::GetFontSize() * 0.5f, 2.0f);
			ImGui::Text("Loading");
			unblock();
			return;
		}

		refreshProjectsFuture_ = {};
	}

	for (const auto& project : *model_.projects)
	{
		ImGui::PushID(project.projectUUID.c_str());

		ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 4.0f);
		ImGui::BeginChild("", ImVec2{ 0, 0 },
						  ImGuiChildFlags_Border | ImGuiChildFlags_AlwaysAutoResize | ImGuiChildFlags_AutoResizeX |
							  ImGuiChildFlags_AutoResizeY);
		ImGui::Text(project.projectName.c_str());
		if (ImGui::Button("Select"))
		{
			model_.selectedProjectUUID = project.projectUUID;
		}

		ImGui::EndChild();
		ImGui::PopStyleVar();
		ImGui::PopID();
	}
	unblock();
}

auto ProjectSelectionView::setModel(Model model) -> void
{

	model_ = std::move(model);
	if (model_.projects == nullptr)
	{
		return;
	}
	if (model_.projects->empty())
	{
		availableProjectString_ = "No projects available!";
	}
	else
	{
		availableProjectString_ = std::format("{} project{} available!", model_.projects->size(),
											  model_.projects->size() < 1 || model_.projects->size() > 1 ? "s" : "");
	}
}

auto ProjectSelectionView::getSelectedProjectUUID() -> std::string
{
	return model_.selectedProjectUUID;
}

auto ProjectSelectionView::projectsAvailable() const -> bool
{
	return model_.projects != nullptr && !model_.projects->empty();
}

auto ProjectSelectionView::requestOngoing() const -> bool
{
	return refreshProjectsFuture_.valid();
}
