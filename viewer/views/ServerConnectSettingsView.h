#pragma once

#include "framework/ModalViewBase.h"

class ServerConnectSettingsView final : public ModalViewBase
{
public:
	ServerConnectSettingsView(ApplicationContext& appContext, const std::string_view name,
							  std::function<void(void)> onSubmitCallback);


// Inherited via ModalViewBase
	auto onDraw() -> void override;
};
