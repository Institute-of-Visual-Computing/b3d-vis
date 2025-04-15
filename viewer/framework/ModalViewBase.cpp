#include "ModalViewBase.h"

#include "ApplicationContext.h"
#include "Color.h"
#include "IdGenerator.h"
#include "Mathematics.h"

#include <format>

ModalViewBase::ModalViewBase(ApplicationContext& applicationContext, const std::string_view name,
							 const ModalType modalType, const ImVec2& minSize)
	: applicationContext_{ &applicationContext }, modalType_{ modalType },
	  id_{ std::format("{}###modal{}", name, IdGenerator::next()) }, name_{ name }, minSize_{ minSize }
{
}

auto ModalViewBase::open() -> void
{
	isOpenRequested_ = true;
	if (onOpenCallback_)
	{
		onOpenCallback_(this);
	}
}

auto ModalViewBase::draw() -> void
{
	if (isOpenRequested_)
	{
		ImGui::OpenPopup(id_.c_str(), ImGuiPopupFlags_AnyPopup);
		isOpenRequested_ = false;
	}
	const auto& brush = applicationContext_->getStyleBrush();

	const auto center = ImGui::GetMainViewport()->GetCenter();
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, Vector2(0.5f, 0.5f));
	ImGui::SetNextWindowSizeConstraints(Vector2{ 400.0f, -1.0f }, Vector2{ 800.0f, -1.0f });
	constexpr auto containerCornerRadius = 8.0f;
	constexpr auto contentCornerRadius = 4.0f;
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, containerCornerRadius);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, containerCornerRadius);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f);
	ImGui::PushStyleColor(ImGuiCol_PopupBg, brush.cardBackgroundFillColorDefaultBrush);
	ImGui::PushStyleColor(ImGuiCol_Border, brush.controlStrokeColorSecondaryBrush);
	if (ImGui::BeginPopupModal(id_.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar))
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Vector2{ 24.0f, 24.0f });
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vector2{ 24.0f, 24.0f });
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, contentCornerRadius);

		ImGui::SetCursorPos(ImGui::GetCursorPos() + Vector2{ 0, 8.0f });
		ImGui::PushFont(applicationContext_->getFontCollection().getTitleFont());
		ImGui::Text(name_.c_str());
		ImGui::PopFont();
		onDraw();
		const auto& style = ImGui::GetStyle();

		const auto position = ImGui::GetCursorScreenPos();
		ImGui::SetCursorPos(ImGui::GetCursorPos() + Vector2{ 0, style.ItemSpacing.y });

		const auto min = position - Vector2{ style.FramePadding.x + 6, 0 };
		const auto max = position +
			Vector2{ style.FramePadding.x + 6, style.FramePadding.y + 10 + style.ItemSpacing.y } +
			ImGui::GetContentRegionAvail();
		ImGui::GetWindowDrawList()->AddRectFilled(
			min, max, brush.solidBackgroundFillColorBaseBrush, containerCornerRadius,
			ImDrawFlags_RoundCornersBottomLeft | ImDrawFlags_RoundCornersBottomRight);
		switch (modalType_)
		{
		case ModalType::ok:
			{
				ImGui::BeginDisabled(isBlocked());
				const auto width = ImGui::GetContentRegionAvail().x;
				if (ui::AccentButton("Ok", Vector2{ width, 0.0f }))
				{
					submit();
					ImGui::CloseCurrentPopup();
				}
				ImGui::EndDisabled();
			}
			break;
		case ModalType::okCancel:
			{
				ImGui::BeginDisabled(isBlocked());
				ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, Vector2{ 8.0f, 8.0f });
				const auto width = (ImGui::GetContentRegionAvail().x - style.ItemSpacing.x) * 0.5f;

				if (ui::AccentButton("Ok", Vector2{ width, 0.0f }))
				{
					submit();
					ImGui::CloseCurrentPopup();
				}
				ImGui::EndDisabled();
				ImGui::SetItemDefaultFocus();
				ImGui::SameLine(0.0f, 8.0f);
				if (ui::Button("Cancel", Vector2{ width, 0.0f }))
				{
					ImGui::CloseCurrentPopup();
				}
				ImGui::PopStyleVar();
			}
			break;
		}

		ImGui::PopStyleVar(3);
		ImGui::EndPopup();
	}
	ImGui::PopStyleColor(2);
	ImGui::PopStyleVar(3);
}

auto ModalViewBase::reset() -> void
{
	block();
}

auto ModalViewBase::submit() -> void
{
	if (!isBlocked())
	{
		if (onSubmitCallback_)
		{
			onSubmitCallback_(this);
		}
	}
}
