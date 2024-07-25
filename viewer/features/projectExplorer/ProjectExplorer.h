#pragma once

#include <memory>

#include "framework/UpdatableComponentBase.h"
#include "ProjectSelectionView.h"

class ApplicationContext;

class ProjectExplorer : public UpdatableComponentBase
{
public:
	ProjectExplorer(ApplicationContext& applicationContext);


private:
	auto update() -> void override;

	std::unique_ptr<ProjectSelectionView> projectSelectionView_{};
};
