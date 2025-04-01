#define IMGUI_DEFINE_MATH_OPERATORS
#include "Project.h"

#include "framework/ApplicationContext.h"

#include "ProjectExplorerView.h"

#include "DebugDrawList.h"
#include "IconsLucide.h"
#include "IdGenerator.h"
#include "ImGuiExtension.h"
#include <imspinner.h>

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
	addNewProjectView_ =
		std::make_unique<AddNewProjectView>(appContext, "Add New Project",
											[&](ModalViewBase* self)
											{
												const auto model = reinterpret_cast<AddNewProjectView*>(self)->model();
												upload = applicationContext_->serverClient_.uploadFileAsync(
													model.sourcePath, model.projectName, uploadFeedback);
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
		});
	deleteProjectView_ = std::make_unique<DeleteProjectView>(
		appContext, "Delete Project", [&]([[maybe_unused]] ModalViewBase* self) {},
		[&](ModalViewBase* self)
		{
			auto view = reinterpret_cast<DeleteProjectView*>(self);
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
	ImGui::BeginChild("", panelSize, ImGuiChildFlags_Border, ImGuiWindowFlags_AlwaysVerticalScrollbar);
	auto pos = ImGui::GetCursorPos();
	const auto widgetStartPosition = ImGui::GetCursorPos();
	const auto itemsCount = items + 1; // add button as last item

	for (auto n = 0; n < itemsCount; n++)
	{
		const auto lastItem = n == items;

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
		if (lastItem)
		{
			static auto isUploadButton = true;

			if (not upload.valid())
			{

				ImGui::PushStyleColor(ImGuiCol_Button,
									  ImGui::ColorConvertFloat4ToU32(ImVec4{ 0.1f, 0.7f, 0.1f, 1.0f }));
				ImGui::SetNextItemAllowOverlap();
				ImGui::PushFont(iconFont);
				if (ImGui::Button(ICON_LC_FILE_UP, itemSize))
				{
					addNewProjectView_->open();
				}
				ImGui::PopFont();
				ImGui::PopStyleColor();
			}
			else
			{
				const auto itemPosition = ImGui::GetCursorPos();
				ImGui::InvisibleButton("##uploading", itemSize);
				const auto textSize = ImGui::CalcTextSize("Uploading");
				ImGui::PushFont(iconFont);
				ImGui::SetNextItemAllowOverlap();
				const auto iconSize = ImGui::CalcTextSize(ICON_LC_FILE_UP);

				const auto height = textSize.y + iconSize.y + ImGui::GetStyle().FramePadding.y;

				ImGui::SetCursorPos(itemPosition +
									ImVec2{ (itemSize.x - iconSize.x) * 0.5f, (itemSize.y - height) * 0.5f });
				ImGui::Text(ICON_LC_FILE_UP);
				ImGui::PopFont();

				const auto text = std::string{ "Uploading" };
				auto approximatedTextSize = textSize;

				ImGui::SetCursorPos(
					itemPosition +
					ImVec2{ (itemSize.x - approximatedTextSize.x) * 0.5f,
							(itemSize.y - height) * 0.5f + iconSize.y + ImGui::GetStyle().FramePadding.y });
				ImGui::Text(text.c_str());
				ImGui::ProgressBar(uploadFeedback.progress / 100.0f);
			}
		}
		else
		{

			const auto itemPosition = ImGui::GetCursorPos();


			if (ImGui::Selectable("", selectedItemIndex == n,
								  ImGuiSelectableFlags_DontClosePopups | ImGuiSelectableFlags_AllowOverlap, itemSize))
			{
				selectedItemIndex = n;
				projectSelected = true;
			}
			const auto isItemHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenOverlappedByItem);
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

			ImGui::SetCursorPos(itemPosition +
								ImVec2{ (itemSize.x - iconSize.x) * 0.5f, (itemSize.y - height) * 0.5f });
			ImGui::Text(icon);
			ImGui::PopFont();


			if (textSize.x - ImGui::GetStyle().FramePadding.x < itemSize.x)
			{
				ImGui::SetCursorPos(
					itemPosition +
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

				ImGui::SetCursorPos(
					itemPosition +
					ImVec2{ (itemSize.x - approximatedTextSize.x) * 0.5f,
							(itemSize.y - height) * 0.5f + iconSize.y + ImGui::GetStyle().FramePadding.y });
				ImGui::Text(text.c_str());
			}
			if (isItemHovered)
			{
				ImGui::SetCursorPos(itemPosition +
									ImVec2{ (itemSize.x) * 0.5f - iconSize.x - ImGui::GetStyle().FramePadding.x * 2 -
												ImGui::GetStyle().ItemSpacing.x * 0.5f,
											(itemSize.y - height) * 0.5f + iconSize.y + dotsTextSize.y +
												ImGui::GetStyle().FramePadding.y });
				ImGui::PushFont(iconFont);
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.7f, 0.7f, 0.3f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.4f, 0.4f, 0.2f, 1.0f });
				ImGui::PushID(n);
				ImGui::SetNextItemAllowOverlap();
				if (ImGui::Button(ICON_LC_PENCIL) || ImGui::IsItemClicked())
				{
					editProjectView_->setProjectIndex(n);
					editProjectView_->open();
				}
				ImGui::PopStyleColor(2);
				ImGui::PopFont();
				ImGui::SetItemTooltip("Edit Project Name");
				ImGui::SameLine();
				ImGui::PushFont(iconFont);
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.1f, 1.0f });
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.4f, 0.1f, 0.1f, 1.0f });
				if (ImGui::Button(ICON_LC_TRASH_2) || ImGui::IsItemClicked())
				{
					deleteProjectView_->setProjectIndex(n);
					deleteProjectView_->open();
				}
				ImGui::PopID();
				ImGui::PopFont();
				ImGui::SetItemTooltip("Delete Project");
				ImGui::PopStyleColor(2);
			}
		}
		ImGui::PopID();
		ImGui::PopStyleVar();
		ImGui::EndChild();
		ImGui::PopStyleVar();
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
	const auto availableWidth = ImGui::GetContentRegionAvail().x;

	const auto scaleFactor = ImGui::GetWindowDpiScale();
	const auto isConnectedToAnyServer = applicationContext_->serverClient_.getLastServerStatusState().health ==
		b3d::tools::project::ServerHealthState::ok;
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
	[[maybe_unused]] const auto switchServerPressed = ImGui::Button(ICON_LC_ARROW_RIGHT_LEFT);
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
		 applicationContext_->serverClient_.getLastServerStatusState().health ==
			 b3d::tools::project::ServerHealthState::testing))
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
		using namespace std::chrono_literals;
		if (upload.valid() and upload.wait_for(0s) == std::future_status::ready)
		{
			const auto uploadResult = upload.get();
			if (uploadResult.state == b3d::tools::project::UploadState::ok)
			{
				model_.projects->push_back(b3d::tools::project::Project{ .projectName = uploadResult.projectName });
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

			if (ImGui::TreeNodeEx("Requests", ImGuiTreeNodeFlags_DefaultOpen))
			{


				for (auto i = 0; i < project.requests.size(); i++)
				{
					ImGui::PushID(i);
					const auto& request = project.requests[i];
					ImGui::SetNextItemAllowOverlap();

					if (loadAndShowFileFuturePerRequest[i].loadAndShowFileFuture.valid())
					{
						loadAndShowFileFuturePerRequest[i].blocked = true;
						if (loadAndShowFileFuturePerRequest[i].loadAndShowFileFuture.wait_for(
								std::chrono::seconds(0)) == std::future_status::ready)
						{
							loadAndShowFileFuturePerRequest[i].loadAndShowFileFuture.get();
							loadAndShowFileFuturePerRequest[i].blocked = false;
						}
					}

					const auto isItemEnabled =
						(request.result.wasSuccess() or loadAndShowFileFuturePerRequest[i].blocked) and
						request.result.nanoResult.fileAvailable;

					ImGui::BeginDisabled(!isItemEnabled);
					if (ImGui::Selectable(request.uuid.c_str(), selectedRequest == i))
					{
						selectedRequest = i;
						loadAndShowFileFuturePerRequest[i].loadAndShowFileFuture =
							loadAndShowFunction_(request.result.nanoResult.resultFile);
					}
					if (ImGui::IsItemHovered())
					{
						const auto box = owl::common::box3f{ { static_cast<float>(request.subRegion.lower.x),
															   static_cast<float>(request.subRegion.lower.y),
															   static_cast<float>(request.subRegion.lower.z) },

															 { static_cast<float>(request.subRegion.upper.x),
															   static_cast<float>(request.subRegion.upper.y),
															   static_cast<float>(request.subRegion.upper.z) } };


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
							  1.0f -
								  blinkIntensity * blinkIntensity * blinkIntensity * blinkIntensity * blinkIntensity },
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
								"File couldn't be loaded! It might be corrupted or an error on server has occured.");
							ImGui::PopTextWrapPos();
							ImGui::EndTooltip();
						}
					}
					const auto selectableSize = ImGui::GetItemRectSize();
					const auto av = ImGui::GetContentRegionAvail();


					const auto viewIconSize = ImGui::CalcTextSize(ICON_LC_VIEW).x;
					const auto detailIconSize = ImGui::CalcTextSize(ICON_LC_VIEW).x;
					constexpr auto spinnerThickness = 2.0f;
					const auto spinnerRadius =
						(selectableSize.y - spinnerThickness) / 2 - ImGui::GetStyle().FramePadding.y;
					const auto spinnerWidth = spinnerRadius * 2;

					const auto framePadding = ImGui::GetStyle().FramePadding.x * 4;
					ImGui::SameLine(selectableSize.x - spinnerWidth - framePadding - detailIconSize - framePadding -
									viewIconSize - framePadding);
					if (loadAndShowFileFuturePerRequest[i].blocked)
					{
						ImSpinner::SpinnerRotateSegments("request_loading_spinner", spinnerRadius, spinnerThickness);
					}


					ImGui::SameLine(selectableSize.x - framePadding - detailIconSize - viewIconSize - framePadding);
					if (ImGui::SmallButton(ICON_LC_VIEW))
					{
						// applicationContext_.
					}
					if (ImGui::BeginItemTooltip())
					{
						ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
						ImGui::TextWrapped("Move the camera view to the selected sub-volume.");
						ImGui::PopTextWrapPos();
						ImGui::EndTooltip();
					}
					ImGui::SameLine(selectableSize.x - framePadding - detailIconSize);
					if (ImGui::SmallButton(ICON_LC_SCROLL_TEXT))
					{
						// applicationContext_.
						parameterSummaryView_->setSofiaParams(request.sofiaParameters);
						parameterSummaryView_->open();
					}
					if (ImGui::BeginItemTooltip())
					{
						ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
						ImGui::Text("Show SoFiA parameters applied for a given request.");
						ImGui::PopTextWrapPos();
						ImGui::EndTooltip();
					}

					ImGui::EndDisabled();

					ImGui::PopID();
				}
				ImGui::TreePop();
			}

			if (applicationContext_->isDevelopmentModeEnabled)
			{
				ImGui::BeginUnderDevelopmentScope();

				constexpr auto tableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
					ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit;

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
				ImGui::EndUnderDevelopmentScope();
			}
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
