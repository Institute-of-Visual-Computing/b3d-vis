#include "ModalViewBase.h"

#include "IdGenerator.h"

#include <format>

ModalViewBase::ModalViewBase(ApplicationContext& appContext, const std::string_view name,
							 const ModalType modalType = ModalType::okCancel)
	: modalType_{ modalType }, id_{ std::format("{}###modal{}", name, IdGenerator::next()) }
{
}

auto ModalViewBase::open() -> void
{
	isOpenRequested_ = true;
}

auto ModalViewBase::draw() -> void
{
	if (isOpenRequested_)
	{
		ImGui::OpenPopup(id_.c_str(), ImGuiPopupFlags_AnyPopup);
		isOpenRequested_ = false;
	}
	ImVec2 center = ImGui::GetMainViewport()->GetCenter();
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
	//ImGui::OpenPopup(id_.c_str());
	if (ImGui::BeginPopupModal(id_.c_str(), NULL, ImGuiWindowFlags_AlwaysAutoResize))
	{

		onDraw();

		switch (modalType_)
		{
		case ModalType::ok:
			ImGui::BeginDisabled(isBlocked());
			if (ImGui::Button("Ok"))
			{
				submit();
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndDisabled();
			break;
		case ModalType::okCancel:
			ImGui::BeginDisabled(isBlocked());
			if (ImGui::Button("Ok"))
			{
				submit();
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndDisabled();
			ImGui::SetItemDefaultFocus();
			ImGui::SameLine();
			if (ImGui::Button("Cancel"))
			{
				ImGui::CloseCurrentPopup();
			}
			break;
		default:
			break;
		}

		ImGui::EndPopup();
	}
}

auto ModalViewBase::reset() -> void
{
	block();
}

auto ModalViewBase::submit() -> void
{
	if (!isBlocked())
	{
		onSubmitCallback_();
	}
}
