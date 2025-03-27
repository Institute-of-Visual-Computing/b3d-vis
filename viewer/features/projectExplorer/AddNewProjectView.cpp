#include "AddNewProjectView.h"
#include "imgui_stdlib.h"


namespace
{
	auto fitsFileValid(const std::filesystem::path& file) -> bool
	{
		std::filesystem::exists(file);
		// TODO: validate axis for example
		return true;
	}
} // namespace

AddNewProjectView::AddNewProjectView(ApplicationContext& appContext, const std::string_view name,
									 const std::function<void(ModalViewBase*)>& onSubmitCallback)
	: ModalViewBase(appContext, name, ModalType::okCancel,
					ImVec2(400 * ImGui::GetFontSize(), 100 * ImGui::GetFontSize()))
{
	setOnSubmit(onSubmitCallback);
}

auto AddNewProjectView::onDraw() -> void
{
	auto path = model_.sourcePath.generic_string();

	ImGui::InputText("Project Name", &model_.projectName);
	ImGui::InputText("FITS File Path", &path);

	model_.sourcePath = path;

	if (not model_.projectName.empty() and fitsFileValid(path))
	{
		unblock();
	}
}
