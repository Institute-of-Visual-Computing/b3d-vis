#pragma once

#include <memory>

#include "ProjectSelectionView.h"
#include "framework/UpdatableComponentBase.h"

class ApplicationContext;

class ProjectExplorer final : public UpdatableComponentBase
{
public:
	explicit ProjectExplorer(ApplicationContext& applicationContext);


private:
	auto update() -> void override;

	std::unique_ptr<ProjectSelectionView> projectSelectionView_{};
};
