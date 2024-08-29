// ReSharper disable CppInconsistentNaming
#define IMGUI_DEFINE_MATH_OPERATORS
#include "ImGuiExtension.h"
#include <imgui.h>

namespace
{
	auto startHeaderPosition = ImVec2{};
}

auto ImGui::BeginUnderDevelopmentScope() -> void
{
	const auto availableWidth = ImGui::GetContentRegionAvail().x;
	auto& drawList = *ImGui::GetWindowDrawList();
	constexpr auto constructionBarWidth = 30.0f;
	constexpr auto constructionBarHeight = constructionBarWidth * 2;
	const auto position = ImGui::GetWindowPos() + ImGui::GetCursorPos() - ImVec2{ constructionBarWidth, 0 };


	startHeaderPosition = ImGui::GetWindowPos() + ImGui::GetCursorPos();


	const auto underDevelopmentText = "UNDER DEVELOPMENT";
	const auto underDevelopmentTextSize = ImGui::CalcTextSize(underDevelopmentText);

	const auto headerSize = ImVec2{ availableWidth, constructionBarHeight };
	const auto textOffset = (headerSize - underDevelopmentTextSize) * 0.5f;

	ImGui::PushClipRect(startHeaderPosition, startHeaderPosition + ImVec2{ availableWidth, headerSize.y }, true);
	for (auto i = 0; i < static_cast<int>(availableWidth / constructionBarWidth) + 2; i++)
	{
		auto p1 = position + ImVec2{ constructionBarWidth + i * constructionBarWidth, 0 };
		auto p2 = position + ImVec2{ constructionBarHeight + i * constructionBarWidth, constructionBarHeight };
		auto p3 = position + ImVec2{ constructionBarWidth + i * constructionBarWidth, constructionBarHeight };
		auto p4 = position + ImVec2{ i * constructionBarWidth, 0 };
		drawList.AddQuadFilled(p1, p2, p3, p4, (i % 2 == 0) ? IM_COL32(200, 200, 0, 255) : IM_COL32(10, 10, 10, 255));
	}


	drawList.AddRectFilled(position + textOffset, position + textOffset + underDevelopmentTextSize, IM_COL32(200, 200, 200, 220));
	drawList.AddText(position + textOffset, IM_COL32(0, 0, 0, 255), underDevelopmentText);
	ImGui::PopClipRect();
	drawList.AddLine(startHeaderPosition + ImVec2{ 0, headerSize.y }, startHeaderPosition + headerSize,
					 IM_COL32(200, 200, 0, 255), 4);
	ImGui::SetCursorPos(ImGui::GetCursorPos() + ImVec2{ 0, headerSize.y + 4 });
}
auto ImGui::EndUnderDevelopmentScope() -> void
{
	auto& drawList = *ImGui::GetWindowDrawList();
	const auto availableWidth = ImGui::GetContentRegionAvail().x;
	drawList.AddRect(startHeaderPosition,
					 ImGui::GetWindowPos() + ImGui::GetCursorPos() + ImVec2{ availableWidth, 0.0f },
					 IM_COL32(200, 200, 0, 255), 0, 0, 4);
}
