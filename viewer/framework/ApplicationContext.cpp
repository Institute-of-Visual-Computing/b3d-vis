#include "ApplicationContext.h"
#include "DebugDrawList.h"
#include "GizmoHelper.h"
#include "framework/Dockspace.h"
#include "framework/UpdatableComponentBase.h"

#include <algorithm>

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

auto ApplicationContext::addMenuAction(Action action, std::string_view menu, std::string_view label,
									   std::optional<std::string_view> shortcut, std::optional<std::string_view> group,
									   int sortOrderKey) -> void
{
	menuData[std::string{ menu }].addItem(group.value_or(""),
										  MenuItemEntryAction{ sortOrderKey, std::string{ label }, action, shortcut });
}

auto ApplicationContext::addMenuToggleAction(bool& toggleValue, ToggleAction onToggleChanged, std::string_view menu,
											 std::string_view label, std::optional<std::string_view> shortcut,
											 std::optional<std::string_view> group, int sortOrderKey) -> void
{
	menuData[std::string{ menu }].addItem(group.value_or(""),
										  MenuItemEntryAction{ sortOrderKey, std::string{ label },
															   ToggleEntryAction{ &toggleValue, onToggleChanged },
															   shortcut });
}

auto ApplicationContext::addMenuBarTray(Action trayDrawCallback) -> void
{
	trayCallbacks.push_back(trayDrawCallback);
}

auto ApplicationContext::MenuItemEntry::addItem(std::string_view group, MenuItemEntryAction actionEntry) -> void
{
	auto& items = groups[std::string{ group }];

	items.push_back(actionEntry);
	std::sort(items.begin(), items.end(),
			  [](const MenuItemEntryAction& a, const MenuItemEntryAction& b) { return a.sortKey < b.sortKey; });
}
