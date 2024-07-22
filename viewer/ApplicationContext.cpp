#include "ApplicationContext.h"
#include "DebugDrawList.h"
#include "GizmoHelper.h"

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
