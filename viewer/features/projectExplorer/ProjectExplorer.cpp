#include "ProjectExplorer.h"

#include "ProjectSelectionView.h"
#include "framework/ApplicationContext.h"

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
