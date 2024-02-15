#include "OpenFileDialog.h"

#include <array>

#include <imgui.h>

using namespace b3d::renderer::nano;

auto OpenFileDialog::open() -> void
{
	ImGui::OpenPopup("FileSelectDialog");
}
auto OpenFileDialog::gui() -> void
{
	//TODO: only select single file now
	std::filesystem::path selectedFile{};

	const auto center = ImGui::GetMainViewport()->GetCenter();
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

	if (ImGui::BeginPopupModal("FileSelectDialog", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
#ifdef WIN32
		constexpr auto roots = std::array{ "A:/", "B:/", "C:/", "D:/", "E:/", "F:/", "G:/", "H:/", "I:/" };

		for (auto i = 0; i < roots.size(); i++)
		{
			const auto root = std::filesystem::path{ roots[i] };
			if (is_directory(root))
			{
				ImGui::SameLine();
				if (ImGui::Button(roots[i]))
				{
					currentPath_ = root;
				}
			}
		}
#endif
		if (ImGui::BeginListBox("##dirs", ImVec2(ImGui::GetFontSize() * 40, ImGui::GetFontSize() * 16)))
		{
			if (ImGui::Selectable("...", false))
			{
				currentPath_ = currentPath_.parent_path();
			}
			auto i = 0;
			for (auto& dir : std::filesystem::directory_iterator{ currentPath_ })
			{
				i++;
				const auto path = dir.path();
				if (is_directory(path))
				{
					if (ImGui::Selectable(dir.path().string().c_str(), false))
					{
						currentPath_ = path;
					}
				}
				const auto isInFilter = filter_.end() != std::ranges::find(filter_, path.extension());
				if (path.has_extension() && isInFilter)
				{
					ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.1f, 0.9f, 0.1f, 1.0f));
					if (ImGui::Selectable(dir.path().string().c_str(), dir.path() == selectedPath_))
					{
						selectedPath_ = dir.path();
					}
					ImGui::PopStyleColor();
				}
			}
			ImGui::EndListBox();
		}
		if (ImGui::Button("OK", ImVec2(120, 0)))
		{
			if (!selectedPath_.empty() != 0)
			{
				selectedItems_.clear();
				selectedItems_.push_back(selectedPath_);
			}
			ImGui::CloseCurrentPopup();
		}
		ImGui::SetItemDefaultFocus();
		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0)))
		{
			selectedPath_.clear();
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
}
