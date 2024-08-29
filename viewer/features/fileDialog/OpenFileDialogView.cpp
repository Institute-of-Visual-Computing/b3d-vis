#include <array>

#include "OpenFileDialogView.h"

auto OpenFileDialogView::getModel() -> Model
{
	return model_;
}

OpenFileDialogView::OpenFileDialogView(ApplicationContext& applicationContext,
									   std::function<void(ModalViewBase* self)> onOpenCallback,
									   std::function<void(ModalViewBase* self)> onSubmitCallback)
	: ModalViewBase(applicationContext, "File selection", ModalType::okCancel, ImVec2{ 800, 600 })
{	
	setOnOpen(onOpenCallback);
	setOnSubmit(onSubmitCallback);
}

OpenFileDialogView::~OpenFileDialogView() = default;

auto OpenFileDialogView::onDraw() -> void
{
#ifdef WIN32
		constexpr auto roots = std::array{ "A:/", "B:/", "C:/", "D:/", "E:/", "F:/", "G:/", "H:/", "I:/", "J:/", "K:/", "L:/", "M:/", "N:/", "O:/", "P:/", "Q:/", "R:/", "S:/", "T:/", "U:/", "V:/", "W:/", "X:/", "Y:/", "Z:/" };

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
					if (ImGui::Selectable(dir.path().string().c_str(), dir.path() == model_.selectedPath_))
					{
						model_.selectedPath_ = dir.path();
					}
					ImGui::PopStyleColor();
				}
			}
			ImGui::EndListBox();
		}
	unblock();
}

auto OpenFileDialogView::setViewParams(const std::filesystem::path& currentPath, const std::vector<std::string>& filter)-> void
{
	model_.selectedPath_.clear();

	currentPath_ = currentPath;
	filter_ = filter;
}
