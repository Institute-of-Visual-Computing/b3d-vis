#include "AddNewProjectView.h"
#include "Style.h"

#include <FitsTools.h>
#pragma warning(push, 0)
#include <ImGuiFileDialog.h>
#pragma warning(pop)

#include <filesystem>
#include <unordered_map>
namespace
{
	struct FitsFileInfoData
	{
		std::filesystem::path selectedFile;
	};

	auto fitsFileValid(const std::filesystem::path& file) -> bool
	{
		return std::filesystem::exists(file) and b3d::tools::fits::isFitsFile(file);
	}

	std::unordered_map<std::filesystem::path, b3d::tools::fits::FitsHeaderInfo> cachedFitsInfos;

	auto fitsFileInfoPanel(const char*, IGFDUserDatas userData, bool*) -> void
	{
		const auto fitsFileInfo = *reinterpret_cast<FitsFileInfoData*>(userData);
		auto fitsInfo = b3d::tools::fits::FitsHeaderInfo{};
		auto isFitsFile = false;
		if (cachedFitsInfos.contains(fitsFileInfo.selectedFile))
		{
			fitsInfo = cachedFitsInfos[fitsFileInfo.selectedFile];
			isFitsFile = true;
		}
		else
		{
			isFitsFile = b3d::tools::fits::isFitsFile(fitsFileInfo.selectedFile);

			if (isFitsFile)
			{
				fitsInfo = b3d::tools::fits::getFitsHeaderInfo(fitsFileInfo.selectedFile);
				cachedFitsInfos[fitsFileInfo.selectedFile] = fitsInfo;
			}
		}

		if (isFitsFile)
		{
			ImGui::SeparatorText("FITS File Info");

			if (fitsInfo.object.has_value())
			{
				ImGui::LabelText("Object", fitsInfo.object.value().c_str());
			}
			if (fitsInfo.fileCreationDate.has_value())
			{
				ImGui::LabelText("Creation Data", fitsInfo.fileCreationDate.value().c_str());
			}
			if (fitsInfo.observationDate.has_value())
			{
				ImGui::LabelText("Observation Data", fitsInfo.observationDate.value().c_str());
			}
			if (fitsInfo.author.has_value())
			{
				ImGui::LabelText("Author", fitsInfo.author.value().c_str());
			}
			if (fitsInfo.observer.has_value())
			{
				ImGui::LabelText("Observer", fitsInfo.observer.value().c_str());
			}
			if (fitsInfo.originOrganisation.has_value())
			{
				ImGui::SeparatorText("Origin Organisation");
				ImGui::TextWrapped(fitsInfo.originOrganisation.value().c_str());
			}
			if (fitsInfo.comment.has_value())
			{
				ImGui::SeparatorText("Comment");
				ImGui::TextWrapped(fitsInfo.comment.value().c_str());
			}
		}
		//*cantContinue = !isFitsFile;
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

	FitsFileInfoData fitsFileData;
	fitsFileData.selectedFile = std::filesystem::path{ ImGuiFileDialog::Instance()->GetCurrentPath() } /
		ImGuiFileDialog::Instance()->GetCurrentFileName();
	ui::HeadedInputText("Source FITS File Path:", "##FITS File Path", &path);
	if (ui::AccentButton(ICON_LC_SEARCH, Vector2{ ImGui::GetContentRegionAvail().x, 0.0f }))
	{
		IGFD::FileDialogConfig config;
		config.path = ".";
		config.countSelectionMax = 1;
		config.sidePane =
			std::bind(&fitsFileInfoPanel, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		config.sidePaneWidth = 200.0f;
		config.userDatas = &fitsFileData;
		config.flags = ImGuiFileDialogFlags_Modal | ImGuiFileDialogFlags_DisableCreateDirectoryButton |
			ImGuiFileDialogFlags_CaseInsensitiveExtentionFiltering;
		ImGuiFileDialog::Instance()->OpenDialog("ChooseFitsFile", "Select Fits File", ".fits", config);
	}

	const auto& io = ImGui::GetIO();
	if (ImGuiFileDialog::Instance()->Display("ChooseFitsFile", ImGuiWindowFlags_NoCollapse,
											 ImVec2{ io.DisplaySize.x * 0.25f, io.DisplaySize.y * 0.25f }))
	{
		if (ImGuiFileDialog::Instance()->IsOk())
		{
			std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
			path = filePathName;
		}
		ImGuiFileDialog::Instance()->Close();
	}

	model_.sourcePath = path;

	if (fitsFileValid(path))
	{
		unblock();
	}
}
