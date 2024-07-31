#pragma once

#include "framework/ModalViewBase.h"
#include "ServerAddEditView.h"

#include <memory>


class ServerConnectSettingsView final : public ModalViewBase
{
public:
	ServerConnectSettingsView(ApplicationContext& appContext, const std::string_view name,
							  std::function<void(void)> onSubmitCallback);


	// Inherited via ModalViewBase
	auto onDraw() -> void override;

private:
	std::unique_ptr<ServerAddEditView> addEditView_;
};
