#define IMGUI_DEFINE_MATH_OPERATORS
#include "Project.h"

#include "framework/ApplicationContext.h"

#include "ProjectExplorerView.h"

#include <imspinner.h>

#include "DebugDrawList.h"
#include "IconsLucide.h"
#include "IdGenerator.h"
#include "ImGuiExtension.h"
#include "Style.h"

namespace
{
	b3d::tools::project::UploadFeedback uploadFeedback;
	std::future<b3d::tools::project::UploadResult> upload;
	int hoveredProjectIndex = -1;
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
	parameterSummaryView_ = std::make_unique<SofiaParameterSummaryView>(appContext);
	addNewProjectView_ = std::make_unique<AddNewProjectView>(
		appContext, "Add New Project",
		[&](ModalViewBase* self)
		{
			const auto model = reinterpret_cast<AddNewProjectView*>(self)->model();
			upload = applicationContext_->serverClient_.uploadFileAsync(model.sourcePath, uploadFeedback);
		});
	editProjectView_ = std::make_unique<EditProjectView>(
		appContext, "Edit Project",
		[&](ModalViewBase* self)
		{
			auto view = reinterpret_cast<EditProjectView*>(self);
			const auto& project = model_.projects->at(view->projectIndex());
			view->setModel(EditProjectModel{ project.projectName });
		},
		[&](ModalViewBase* self)
		{
			auto view = reinterpret_cast<EditProjectView*>(self);
			auto& project = model_.projects->at(view->projectIndex());
			project.projectName = view->model().projectName;
			changeProjectFuture_ =
				applicationContext_->serverClient_.changeProjectAsync(project.projectUUID, project.projectName);
		});
	deleteProjectView_ = std::make_unique<DeleteProjectView>(
		appContext, "Delete Project", [&]([[maybe_unused]] ModalViewBase* self) {},
		[&](ModalViewBase* self)
		{
			auto view = reinterpret_cast<DeleteProjectView*>(self);
			std::string uuid = model_.projects->at(view->projectIndex()).projectUUID;
			deleteProjectFuture_ = applicationContext_->serverClient_.deleteProjectAsync(uuid);
			model_.projects->erase(model_.projects->begin() + view->projectIndex());
		});
}

ProjectExplorerView::~ProjectExplorerView() = default;

auto ProjectExplorerView::setModel(Model model) -> void
{
	model_ = std::move(model);
}


auto ProjectExplorerView::drawSelectableItemGridPanel(const char* panelId, int& selectedItemIndex, const int items,
													  const std::function<const char*(const int index)>& name,
													  const char* icon, ImFont* iconFont,
													  const std::function<void(const int index)>& popup,
													  const ImVec2 itemSize, const ImVec2 panelSize) -> bool
{
	auto projectSelected{ false };
	const auto& style = ImGui::GetStyle();
	const auto windowVisibleX2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
	ImGui::PushID(panelId);
	ImGui::BeginChild("", panelSize, ImGuiChildFlags_Border | ImGuiChildFlags_FrameStyle);

	auto pos = ImGui::GetCursorPos();
	const auto widgetStartPosition = ImGui::GetCursorPos();
	const auto itemsCount = items;
	for (auto n = 0; n < itemsCount; n++)
	{
		constexpr auto padding = 10;

		ImGui::PushID(n);
		ImGui::SetNextItemAllowOverlap();
		ImGui::SetCursorPos(Vector2(pos.x + padding, pos.y + padding));
		ImGui::BeginChild("", Vector2{ 0, 0 },
						  ImGuiChildFlags_AlwaysAutoResize | ImGuiChildFlags_AutoResizeX | ImGuiChildFlags_AutoResizeY);
		auto alignment = Vector2(0.5f, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, alignment);
		ImGui::PushID(n);

		const auto itemPosition = ImGui::GetCursorPos();

		if (ui::ToggleButton(selectedItemIndex == n, "", itemSize))
		{
			selectedItemIndex = n;
			projectSelected = true;
		}

		if (ImGui::BeginItemTooltip())
		{
			popup(n);
			ImGui::EndTooltip();
		}

		const auto textSize = ImGui::CalcTextSize(name(n));
		auto approximatedTextSize = textSize;
		const auto dotsTextSize = ImGui::CalcTextSize("...");
		ImGui::PushFont(iconFont);
		// ImGui::SetNextItemAllowOverlap();
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

		const auto lastButtonX2 = ImGui::GetItemRectMax().x;
		const auto nextButtonX2 = lastButtonX2 + style.ItemSpacing.x + itemSize.x;
		if (n + 1 < itemsCount && nextButtonX2 < windowVisibleX2)
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
	return projectSelected;
}


auto ProjectExplorerView::onDraw() -> void
{
	const auto isConnectedToAnyServer = applicationContext_->serverClient_.getLastServerStatusState().health ==
		b3d::tools::project::ServerHealthState::ok;
	const auto isAnyProjectAvailable = projectAvailable();

	const auto& brush = ApplicationContext::getStyleBrush();
	constexpr auto containerCornerRadius = 8.0f;
	constexpr auto contentCornerRadius = 4.0f;
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, containerCornerRadius);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, containerCornerRadius);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f);

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vector2{ 24.0f, 24.0f });
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vector2{ 24.0f, 24.0f });
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, contentCornerRadius);

	ImGui::PushStyleColor(ImGuiCol_PopupBg, brush.cardBackgroundFillColorDefaultBrush);
	ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrokeColorSecondaryBrush);

	const auto font = applicationContext_->getFontCollection().getTitleFont();
	ImGui::PushFont(font);
	ImGui::Text("Project Explorer");
	ImGui::PopFont();

	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vector2{ 24.0f, 12.0f });
	ImGui::TextWrapped(
		"This tool lets you connect to the remote or local dataset repository and manage its data cubes.");
	ImGui::TextLinkOpenURL("Learn more about the workflow", "https://github.com/Institute-of-Visual-Computing/b3d-vis");
	ImGui::TextLinkOpenURL("Learn more about how to set up a data repository server",
						   "https://github.com/Institute-of-Visual-Computing/b3d-vis");
	ImGui::PopStyleVar();

	if (not isConnectedToAnyServer)
	{
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.systemFillColorCautionBackgroundBrush);

		ImGui::BeginChild("##is_not_connected_to_server_warning", Vector2{},
						  ImGuiChildFlags_AlwaysAutoResize | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_FrameStyle);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.systemFillColorCautionBrush);
		ImGui::Text(ICON_LC_MESSAGE_SQUARE_WARNING);
		ImGui::PopStyleColor();
		ImGui::SameLine();
		ImGui::TextWrapped("You are not currently connected to any data repository server !To set up the data server, "
						   "navigate to File > Server Connection.");
		ImGui::EndChild();
		ImGui::PopStyleColor();
	}

	if (isConnectedToAnyServer and not isAnyProjectAvailable)
	{
		ImGui::PushStyleColor(ImGuiCol_FrameBg, brush.systemFillColorSuccessBackgroundBrush);

		ImGui::BeginChild("##empty_server_warning", Vector2{},
						  ImGuiChildFlags_AlwaysAutoResize | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_FrameStyle);
		ImGui::PushStyleColor(ImGuiCol_Text, brush.systemFillColorSuccessBrush);
		ImGui::Text(ICON_LC_MESSAGE_SQUARE_WARNING);
		ImGui::PopStyleColor();
		ImGui::SameLine();
		ImGui::TextWrapped(
			"You are connected to the data server, but its repository is empty! Add new dataset by clicking on Add and "
			"select an HI-Datacube FITS file from you local storage and start the upload.");
		ImGui::EndChild();
		ImGui::PopStyleColor();
	}

	ImGui::PopStyleColor(2);
	ImGui::PopStyleVar(6);

	const auto availableWidth = ImGui::GetContentRegionAvail().x;

	const auto scaleFactor = ImGui::GetWindowDpiScale();

	const auto serverNameText = std::format(
		"Server: {}",
		isConnectedToAnyServer ? applicationContext_->serverClient_.getConnectionInfo().name : "Disconnected!");
	const auto textSize = ImGui::CalcTextSize(serverNameText.c_str());

	const auto positionForLoadingPlaceholder = ImGui::GetCursorPos();
	const auto middleSpace = availableWidth - textSize.x;

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

	auto isStillRefreshing = false;
	if (refreshProjectsFuture_.valid() and
		(isConnectedToAnyServer or
		 applicationContext_->serverClient_.getLastServerStatusState().health ==
			 b3d::tools::project::ServerHealthState::testing))
	{
		isStillRefreshing = true;
		if (not(refreshProjectsFuture_.wait_for(std::chrono::seconds(0)) != std::future_status::ready))
		{
			refreshProjectsFuture_ = {};
		}
	}

	if (isStillRefreshing)
	{
		ImSpinner::SpinnerRotateSegments("project_loading_spinner", ImGui::GetFontSize() * 0.5f, 2.0f);
		ImGui::SameLine();
		ImGui::Text("Loading");
	}
	else
	{
		if (ui::Button(ICON_LC_REFRESH_CW))
		{
			refreshProjectsFuture_ = refreshProjectsFunction_();
		}
	}

	ImGui::SetItemTooltip("Refresh");
	ImGui::SameLine(middleSpace);
	ImGui::Text(serverNameText.c_str());
	// TODO: Do we really need this button?
	/*ImGui::SameLine();
	if (ui::Button(ICON_LC_ARROW_RIGHT_LEFT))
	{
	}
	ImGui::SetItemTooltip("Switch Server");*/

	if (applicationContext_->isDevelopmentModeEnabled)
	{
		ImGui::BeginUnderDevelopmentScope();
		if (ui::Button("Load .nvdb manually"))
		{
			showNvdbSelectionModal_();
		}
		ImGui::EndUnderDevelopmentScope();
	}

	if (model_.projects)
	{
		using namespace std::chrono_literals;
		if (upload.valid() and upload.wait_for(0s) == std::future_status::ready)
		{
			const auto uploadResult = upload.get();
			if (uploadResult.state == b3d::tools::project::UploadState::ok)
			{
				if (uploadResult.project.has_value())
				{
					model_.projects->push_back(uploadResult.project.value());
				}
				uploadFeedback.progress = 0;
			}
		}

		const auto projectChanged = drawSelectableItemGridPanel(
			"projects", selectedProjectItemIndex_, static_cast<const int>(model_.projects->size()),
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
			ImVec2{ 100 * scaleFactor, 100 * scaleFactor });


		const auto& style = ImGui::GetStyle();
		const auto addButtonText = ICON_LC_FILE_UP " Add";
		const auto addButtonTextSize = ImGui::CalcTextSize(addButtonText, NULL, true);
		const auto addButtonSize = ImGui::CalcItemSize(Vector2{}, addButtonTextSize.x + style.FramePadding.x * 2.0f,
													   addButtonTextSize.y + style.FramePadding.y * 2.0f);

		if (not upload.valid())
		{


			ImGui::SetNextItemAllowOverlap();
			if (ui::AccentButton(addButtonText, addButtonSize))
			{
				addNewProjectView_->open();
			}
			if (ImGui::BeginItemTooltip())
			{
				ImGui::Text("Create and Upload new dataset");
				ImGui::EndTooltip();
			}
		}
		else
		{
			const auto itemPosition = Vector2{ ImGui::GetCursorPos() };
			ImGui::InvisibleButton("##uploading", addButtonSize);
			ImGui::SetNextItemAllowOverlap();

			ImGui::SetCursorPos(itemPosition);
			ImGui::ProgressBar(uploadFeedback.progress / 100.0f, addButtonSize, "");
			const auto bars = 12u;
			ImGui::SetCursorPos(itemPosition + Vector2{ addButtonSize.x * 0.25f, 0 });
			ImSpinner::SpinnerBarsScaleMiddle("uploading_new_file", ImGui::GetFontSize() * 0.5f / bars,
											  ImSpinner::white, 2.8f, bars);
		}

		const auto isValidSelection = selectedProjectItemIndex_ >= 0;
		ImGui::BeginDisabled(not isValidSelection);
		{
			ImGui::PushID(selectedProjectItemIndex_);
			const auto emptySpace = ImGui::GetContentRegionAvail().x -
				(ImGui::CalcTextSize(ICON_LC_PENCIL " Edit").x + ImGui::CalcTextSize(ICON_LC_TRASH_2 " Delete").x +
				 style.FramePadding.x * 4.0f + addButtonSize.x + style.ItemSpacing.x);

			ImGui::SameLine(0, emptySpace);
			if (ui::Button(ICON_LC_PENCIL " Edit"))
			{
				editProjectView_->setProjectIndex(selectedProjectItemIndex_);
				editProjectView_->open();
			}
			ImGui::SetItemTooltip("Edit Project Name");
			ImGui::SameLine();

			if (ui::Button(ICON_LC_TRASH_2 " Delete"))
			{
				deleteProjectView_->setProjectIndex(selectedProjectItemIndex_);
				deleteProjectView_->open();
			}
			ImGui::SetItemTooltip("Delete Project");
			ImGui::PopID();
		}
		ImGui::EndDisabled();

		if (selectedProjectItemIndex_ >= 0)
		{
			struct RequestLoad
			{
				bool blocked;
				std::shared_future<void> loadAndShowFileFuture;
			};

			constexpr auto defaultVolumeDataRequest = 0; // TODO: always load the first request at startup


			static std::vector<RequestLoad> loadAndShowFileFuturePerRequest{};
			if (projectChanged)
			{
				applicationContext_->selectedProject_ = model_.projects->at(selectedProjectItemIndex_);
				loadAndShowFileFuturePerRequest.resize(applicationContext_->selectedProject_->requests.size());
				loadAndShowFileFuturePerRequest[defaultVolumeDataRequest].loadAndShowFileFuture = loadAndShowFunction_(
					applicationContext_->selectedProject_->requests[0].result.nanoResult.resultFile);
			}
			const auto& project = model_.projects->at(selectedProjectItemIndex_);

			if (project.requests.size() != loadAndShowFileFuturePerRequest.size())
			{
				static std::vector<RequestLoad> tmp{};
				tmp.resize(project.requests.size());
				for (auto i = 0; i < loadAndShowFileFuturePerRequest.size(); i++)
				{
					tmp[i] = loadAndShowFileFuturePerRequest[i];
				}
				loadAndShowFileFuturePerRequest = tmp;
			}
			ImGui::Separator();
			ImGui::Text(std::format("Selected Dataset: {}", project.projectName).c_str());
			ImGui::Separator();
			if (applicationContext_->isDevelopmentModeEnabled)
			{
				ImGui::BeginUnderDevelopmentScope();
				if (ImGui::CollapsingHeader(project.fitsOriginFileName.c_str()))
				{
					ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
					ImGui::BeginChild("Properties", ImVec2(0, 260 * scaleFactor), ImGuiChildFlags_Border, window_flags);

					for (auto i = 0; i < project.fitsOriginProperties.axisTypes.size(); i++)
					{
						ImGui::LabelText(std::format("Axis {}", i).c_str(),
										 project.fitsOriginProperties.axisTypes[i].c_str());
					}

					ImGui::EndChild();
					ImGui::PopStyleVar();
				}
				ImGui::EndUnderDevelopmentScope();
			}

			static auto selectedRequest = defaultVolumeDataRequest;
			ImGui::BeginChild("##requests", ImGui::GetContentRegionAvail(),
							  ImGuiChildFlags_Border | ImGuiChildFlags_FrameStyle);
			/*ImGui::GetForegroundDrawList()->PushClipRect(
				ImGui::GetCursorPos(), Vector2{ ImGui::GetCursorPos() } + Vector2{ ImGui::GetContentRegionAvail() },
				false);*/
			for (auto i = 0; i < project.requests.size(); i++)
			{
				ImGui::PushID(i);
				const auto& request = project.requests[i];
				ImGui::SetNextItemAllowOverlap();

				if (loadAndShowFileFuturePerRequest[i].loadAndShowFileFuture.valid())
				{
					loadAndShowFileFuturePerRequest[i].blocked = true;
					if (loadAndShowFileFuturePerRequest[i].loadAndShowFileFuture.wait_for(std::chrono::seconds(0)) ==
						std::future_status::ready)
					{
						loadAndShowFileFuturePerRequest[i].loadAndShowFileFuture.get();
						loadAndShowFileFuturePerRequest[i].blocked = false;
					}
				}

				const auto isItemEnabled =
					(request.result.wasSuccess() or loadAndShowFileFuturePerRequest[i].blocked) and
					request.result.nanoResult.fileAvailable;

				ImGui::BeginDisabled(!isItemEnabled);

				const auto position = Vector2{ ImGui::GetCursorPos() };
				if (ui::Selectable("", selectedRequest == i, 0, Vector2{ 0.0f, ImGui::GetTextLineHeight() * 2.0f }))
				{
					selectedRequest = i;
					loadAndShowFileFuturePerRequest[i].loadAndShowFileFuture =
						loadAndShowFunction_(request.result.nanoResult.resultFile);
				}
				const auto nextItem = ImGui::GetCursorPos();
				if (ImGui::IsItemHovered())
				{
					const auto box = owl::common::box3f{
						{ static_cast<float>(request.subRegion.lower.x), static_cast<float>(request.subRegion.lower.y),
						  static_cast<float>(request.subRegion.lower.z) },

						{ static_cast<float>(request.subRegion.upper.x), static_cast<float>(request.subRegion.upper.y),
						  static_cast<float>(request.subRegion.upper.z) }
					};


					const auto originalBoxSize = owl::vec3f{
						static_cast<float>(-(project.fitsOriginProperties.axisDimensions[0] - 1)),
						static_cast<float>(-(project.fitsOriginProperties.axisDimensions[1] - 1)),
						static_cast<float>(project.fitsOriginProperties.axisDimensions[2] - 1),
					};

					auto boxTranslate = box.size() / 2.0f;
					boxTranslate.z *= -1.0f;
					boxTranslate += owl::vec3f{ box.lower.x, box.lower.y, -box.lower.z };


					constexpr auto blinkFrequency = 10.0f;
					const auto blinkIntensity =
						0.5f + 0.5f * glm::sin(ImGui::GetCurrentContext()->HoveredIdTimer * blinkFrequency);
					applicationContext_->getDrawList()->drawBox(
						volumeTransform_.p / 2, originalBoxSize / 2.0f + boxTranslate, box.size(),
						{ 1.0, 0.0, 0.0,
						  1.0f - blinkIntensity * blinkIntensity * blinkIntensity * blinkIntensity * blinkIntensity },
						volumeTransform_.l);
				}
				if (!request.result.wasSuccess())
				{
					if (ImGui::BeginItemTooltip())
					{
						ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
						ImGui::TextWrapped(std::format("Failed with error code {} and message: {}",
													   request.result.sofiaResult.returnCode,
													   request.result.sofiaResult.message)
											   .c_str());
						ImGui::PopTextWrapPos();
						ImGui::EndTooltip();
					}
				}
				else if (!request.result.nanoResult.fileAvailable)
				{
					if (ImGui::BeginItemTooltip())
					{
						ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
						ImGui::TextWrapped(
							"File couldn't be loaded! It might be corrupted or an error on server has occurred.");
						ImGui::PopTextWrapPos();
						ImGui::EndTooltip();
					}
				}
				ImGui::SetCursorPos(position + Vector2{ 8.0f, 0.0f } +
									Vector2{ 0.0f, ImGui::GetTextLineHeight() * 0.5f });
				ImGui::Text(request.uuid.c_str());
				ImGui::SetCursorPos(nextItem);
				const auto selectableSize = ImGui::GetItemRectSize();
				const auto av = ImGui::GetContentRegionAvail();

				// TODO (Anton)
				/*const auto viewIconSize = ImGui::CalcTextSize(ICON_LC_VIEW).x;
				const auto detailIconSize = ImGui::CalcTextSize(ICON_LC_VIEW).x;
				constexpr auto spinnerThickness = 2.0f;
				const auto spinnerRadius =
					(selectableSize.y - spinnerThickness) / 2 - ImGui::GetStyle().FramePadding.y * 2;
				const auto spinnerWidth = spinnerRadius * 2;

				const auto framePadding = ImGui::GetStyle().FramePadding.x * 4;*/

				/*if (loadAndShowFileFuturePerRequest[i].blocked)
				{
					ImGui::SameLine(selectableSize.x - spinnerWidth - framePadding - detailIconSize - framePadding -
									viewIconSize - framePadding);
					ImSpinner::SpinnerRotateSegments("request_loading_spinner", spinnerRadius, spinnerThickness);
				}*/

				ImGui::EndDisabled();

				ImGui::PopID();
			}
			// ImGui::GetForegroundDrawList()->PopClipRect();
			ImGui::EndChild();
		}
	}
	parameterSummaryView_->draw();
	addNewProjectView_->draw();
	editProjectView_->draw();
	deleteProjectView_->draw();
}

auto ProjectExplorerView::projectAvailable() const -> bool
{
	return model_.projects != nullptr;
}
