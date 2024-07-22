#include "ModalViewBase.h"

ModalViewBase::ModalViewBase(ApplicationContext& appContext, const std::string_view name,
							 const ModalType modalType = ModalType::okCancel)
	: WindowViewBase(appContext, name, WindowFlagBits::noCollapse | WindowFlagBits::noClose | WindowFlagBits::noDocking),
	  modalType_{ modalType }
{
	close();
}

auto ModalViewBase::draw() -> void
{
	beginDraw();
	onDraw();

	switch (modalType_)
	{
	case ModalType::ok:
		ImGui::BeginDisabled(isBlocked());
		if (ImGui::Button("Ok"))
		{
			submit();
			requestClose();
		}
		ImGui::EndDisabled();
		break;
	case ModalType::okCancel:
		ImGui::BeginDisabled(isBlocked());
		if (ImGui::Button("Ok"))
		{
			submit();
			requestClose();
		}
		ImGui::EndDisabled();
		ImGui::SameLine();
		if (ImGui::Button("Cancel"))
		{
			requestClose();
		}
		break;
	default:
		break;
	}

	endDraw();

	if (closeRequested_)
	{
		close();
	}
}

auto ModalViewBase::reset() -> void
{
	block();
	closeRequested_ = false;
}

auto ModalViewBase::submit() -> void
{
	if (!isBlocked())
	{
		onSubmitCallback_();
	}

	requestClose();
}
