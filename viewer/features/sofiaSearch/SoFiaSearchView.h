#pragma once

#include "framework/DockableWindowViewBase.h"

class SoFiaSearchView  final : public DockableWindowViewBase
{
public:
	struct Model
	{
	};

	SoFiaSearchView(ApplicationContext& appContext, Dockspace* dockspace);
	~SoFiaSearchView() override;
	
	auto setModel(Model model) -> void;
	auto getModel() const -> const Model&;

private:
	auto onDraw() -> void override;

	Model model_;
};
