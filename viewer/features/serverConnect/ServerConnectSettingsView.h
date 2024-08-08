#pragma once

#include "ServerAddEditView.h"
#include "framework/ModalViewBase.h"

#include <memory>


class ServerConnectSettingsView final : public ModalViewBase
{
public:
	ServerConnectSettingsView(ApplicationContext& appContext, const std::string_view name,
							  const std::function<void(ModalViewBase*)>& onSubmitCallback);


	// Inherited via ModalViewBase
	auto onDraw() -> void override;

private:
	std::unique_ptr<ServerAddEditView> addServerView_;
	std::unique_ptr<ServerAddEditView> editServerView_;

	int selectedItem_{ -1 };

	[[nodiscard]] auto isServerSelected() const -> bool
	{
		return selectedItem_ != -1;
	}

	auto testServerStatus() -> void;

	enum class ServerStatusState
	{
		ok,
		unreachable,
		unknown,
		testing
	};

	ServerStatusState selectedServerStatus_{ServerStatusState::unknown};

	auto resetServerStatus() -> void;
	
};
