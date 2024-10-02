#pragma once

#include "ServerClient.h"
#include "framework/ModalViewBase.h"

class ServerAddEditView final : public ModalViewBase
{
public:
	ServerAddEditView(ApplicationContext& applicationContext, const std::string_view name,
					  std::function<void(ModalViewBase* self)> onOpenCallback,
					  std::function<void(ModalViewBase* self)> onSubmitCallback);

	auto onDraw() -> void override;

private:
	b3d::tools::project::ServerConnectionDescription model_{};

public:
	auto setModel(const b3d::tools::project::ServerConnectionDescription& model) -> void
	{
		model_ = model;
	}

	[[nodiscard]] auto model() const -> const b3d::tools::project::ServerConnectionDescription&
	{
		return model_;
	}
};
