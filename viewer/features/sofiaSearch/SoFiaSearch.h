#pragma once

#include <memory>

#include "framework/UpdatableComponentBase.h"

class SoFiaSearchView;

class SoFiaSearch final : public UpdatableComponentBase
{
public:
	SoFiaSearch(ApplicationContext& applicationContext);
	~SoFiaSearch() override;

	auto update() -> void override;
private:
	bool showSearchWindow_{ true };

	std::unique_ptr<SoFiaSearchView> sofiaSearchView_;
};
