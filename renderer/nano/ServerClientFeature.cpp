#pragma once

#include "ServerClientFeature.h"

#include "NanoRenderer.h"
#include "owl/helper/cuda.h"

namespace
{
	auto loadNvdbGUIFile(b3d::tools::server_client::Client& client, std::string fileGUID, b3d::renderer::nano::RuntimeDataSet& runtimeDataset) -> void
	{
		auto filePath = client.getFile(fileGUID);
		if (filePath.empty())
		{
			return;
		}
		cudaStream_t stream;
		OWL_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
		runtimeDataset.addNanoVdb(filePath, stream);
		cudaStreamDestroy(stream);
	}
}

b3d::renderer::ServerClientFeature::ServerClientFeature(const std::string& name, b3d::renderer::NanoRenderer* renderer)
	: RenderFeature(name), renderer_(renderer)
{
}

void b3d::renderer::ServerClientFeature::onInitialize()
{
	RenderFeature::onInitialize();
}

auto b3d::renderer::ServerClientFeature::gui() -> void
{
	ImGui::SeparatorText("Server");

	if (requestOngoing())
	{
		ImGui::Text("Busy");

		using namespace std::chrono_literals;
		if (loadingRequest && loadingRequest->valid())
		{
			const auto ret = loadingRequest->wait_for(0s);
			if (ret == std::future_status::ready)
			{
				loadingRequest.reset();
			}
		}
		else
		{
			loadingRequest.reset();
		}

		if (projectLoadingRequest && projectLoadingRequest->valid())
		{
			const auto ret = projectLoadingRequest->wait_for(0s);
			if (ret == std::future_status::ready)
			{
				projects = projectLoadingRequest->get();
				projectLoadingRequest.reset();
			}
		}
		else
		{
			projectLoadingRequest.reset();
		}

		ImGui::Text("Processing request.");
	}
	else
	{
		if (ImGui::Button("Refresh Projects"))
		{
			auto req = std::async(std::launch::async, &tools::server_client::Client::getProjects, serverClient);
			projectLoadingRequest = std::make_unique<std::future<std::vector<tools::projectexplorer::Project>>>(std::move(req));
		}
		ImGui::Spacing();
		ImGui::Separator();
		if (!projects.empty())
		{
			ImGui::SeparatorText(std::format("{} Project/s", projects.size()).c_str());
		}

		for (const auto& project : projects)
		{
			if (project.requests.empty())
			{
				ImGui::Text(project.projectName.c_str());
			}
			else
			{
				if (ImGui::TreeNode(project.projectName.c_str()))
				{
					
					ImGui::SameLine();
					if (selectedProjectUUID_ == project.projectUUID)
					{
						ImGui::BeginDisabled(true);
						ImGui::Button("Selectd");
						ImGui::EndDisabled();
					}
					else
					{
						if (ImGui::Button("Select"))
						{
							
							selectProject(project);
						}
					}

					if (ImGui::TreeNode("Properties"))
					{
						// ImGui::LabelText("Axis count", std::to_string(project.fitsOriginProperties.axisCount).c_str());
						static ImGuiTableFlags flags = ImGuiTableFlags_RowBg;

						ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
						ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
						ImGui::BeginChild("ChildR",
										  ImVec2(-FLT_MIN, ImGui::GetTextLineHeightWithSpacing() * 5),
										  ImGuiChildFlags_Border, window_flags);

						ImGui::LabelText("File name" ,project.fitsOriginFileName.c_str());
						ImGui::Spacing();
						for (auto i = 0; i < project.fitsOriginProperties.axisCount; i++)
						{
							ImGui::Text(std::format("Axis {}", i + 1).c_str());
							ImGui::SameLine();
							ImGui::Text(std::to_string(project.fitsOriginProperties.axisDimensions[i]).c_str());
							ImGui::SameLine();
							ImGui::Text(project.fitsOriginProperties.axisTypes[i].c_str());

						}
						
						ImGui::EndChild();
						ImGui::PopStyleVar();
						ImGui::TreePop();
					}

					if (project.requests.size() > 1)
					{
						if (ImGui::TreeNode("Requests"))
						{
							for (auto i = 1; i < project.requests.size(); i++)
							{
								const auto& request = project.requests[i];
							
								if (request.result.wasSuccess())
								{
									ImGui::Text("Request");
									if (!request.result.nvdb.fileResultGUID.empty())
									{
										ImGui::SameLine();
										if (ImGui::Button("Load"))
										{
											loadNvdb(request.result.nvdb.fileResultGUID);
										}
									}
								}
								else
								{
									
								}
							}
							ImGui::TreePop();
						}
					}
					ImGui::TreePop();
				}
			}
		}
	}
}

auto b3d::renderer::ServerClientFeature::getParamsData() -> ParamsData
{
	
	return {};
}

auto b3d::renderer::ServerClientFeature::hasGui() const -> bool
{
	return true;
}

auto b3d::renderer::ServerClientFeature::requestOngoing() -> bool
{
	if (loadingRequest || projectLoadingRequest)
	{
		return true;
	}

	return false;
}

auto b3d::renderer::ServerClientFeature::selectProject(const tools::projectexplorer::Project& project) -> void

{
	selectedProjectUUID_ = project.projectUUID;
	loadNvdb(project.requests[0].result.nvdb.fileResultGUID);
}

auto b3d::renderer::ServerClientFeature::loadNvdb(const std::string& nvdbFileGuid) -> void
{
	auto req = std::async(std::launch::async, loadNvdbGUIFile, std::ref(serverClient), nvdbFileGuid,
						  std::ref(renderer_->runtimeDataSet_));

	loadingRequest = std::make_unique<std::future<void>>(std::move(req));
}
