#include "ApplicationContext.h"
#include "DebugDrawList.h"
#include "GizmoHelper.h"
#include "framework/UpdatableComponentBase.h"
#include "framework/Dockspace.h"

ApplicationContext::ApplicationContext()
{
	mainDockspace_ = std::make_unique<Dockspace>();
}

auto ApplicationContext::setExternalDrawLists(std::shared_ptr<DebugDrawList> debugDrawList,
											  std::shared_ptr<GizmoHelper> gizmoHelper) -> void
{
	debugDrawList_ = debugDrawList;
	gizmoHelper_ = gizmoHelper;
}

auto ApplicationContext::getGizmoHelper() const -> std::shared_ptr<GizmoHelper>
{
	return gizmoHelper_;
}

auto ApplicationContext::getDrawList() const -> std::shared_ptr<DebugDrawList>
{
	return debugDrawList_;
}

auto ApplicationContext::getMainDockspace() -> Dockspace*
{
	return mainDockspace_.get();
}

auto ApplicationContext::addUpdatableComponent(UpdatableComponentBase* component) -> void
{
	updatableComponents_.push_back(component);
}

auto ApplicationContext::addRendererExtensionComponent(RendererExtensionBase* component) -> void
{
	rendererExtensions_.push_back(component);
}
