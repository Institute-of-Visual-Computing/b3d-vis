#include "ModalViewBase.h"

#include "IdGenerator.h"

#include <format>

ModalViewBase::ModalViewBase(ApplicationContext& applicationContext, const std::string_view name,
							 const ModalType modalType, const ImVec2& minSize)
	: applicationContext_{ &applicationContext }, modalType_{ modalType },
	  id_{ std::format("{}###modal{}", name, IdGenerator::next()) }, minSize_{ minSize }
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
	ImVec2 center = ImGui::GetMainViewport()->GetCenter();
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
	ImGui::SetNextWindowSizeConstraints(minSize_, ImVec2{ INFINITY, -1.0 });
	// ImGui::OpenPopup(id_.c_str());
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
		if (onSubmitCallback_)
		{
			onSubmitCallback_(this);
		}
	}
}
