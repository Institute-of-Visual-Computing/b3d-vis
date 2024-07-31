#pragma once

#include "framework/ModalViewBase.h"
#include "framework/UpdatableComponentBase.h"

class ServerAddEditView final : public ModalViewBase, public UpdatableComponentBase
{
public:
	ServerAddEditView(ApplicationContext& applicationContext, const std::string_view name,
							  std::function<void(void)> onSubmitCallback);


	// Inherited via ModalViewBase
	auto onDraw() -> void override;

	// Inherited via UpdatableComponentBase
	auto update() -> void override;
};
