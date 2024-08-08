#include "ProjectExplorer.h"

#include "framework/ApplicationContext.h"
#include "ProjectSelectionView.h"

ProjectExplorer::ProjectExplorer(ApplicationContext& applicationContext) : UpdatableComponentBase{applicationContext}
{
	projectSelectionView_ = std::make_unique<ProjectSelectionView>(applicationContext);
	applicationContext.addMenuAction([&]() { projectSelectionView_->open();
		}, "Program", "Open Project...");

	applicationContext.addUpdatableComponent(this);
}

auto ProjectExplorer::update() -> void
{
	projectSelectionView_->draw();
}
