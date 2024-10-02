#include "ApplicationContext.h"
#include "DebugDrawList.h"
#include "GizmoHelper.h"
#include "framework/Dockspace.h"

#include <algorithm>

ApplicationContext::ApplicationContext()
{
	mainDockspace_ = std::make_unique<Dockspace>();
	profiler_ = std::make_unique<b3d::profiler::Profiler>();
}

auto ApplicationContext::setExternalDrawLists(const std::shared_ptr<DebugDrawList>& debugDrawList,
											  const std::shared_ptr<GizmoHelper>& gizmoHelper) -> void
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

auto ApplicationContext::getMainDockspace() const -> Dockspace*
{
	return mainDockspace_.get();
}

auto ApplicationContext::addUpdatableComponent(UpdatableComponentBase* component) -> void
{
	updatableComponents_.push_back(component);
}

auto ApplicationContext::removeUpdatableComponent(UpdatableComponentBase* component) -> void
{
	// std::remove(updatableComponents_.begin(), updatableComponents_.end(), component);
	std::erase(updatableComponents_, component);
}

auto ApplicationContext::addRendererExtensionComponent(RendererExtensionBase* component) -> void
{
	rendererExtensions_.push_back(component);
}

auto ApplicationContext::addMenuAction(Action action, const std::string_view menu, const std::string_view label,
									   const std::optional<std::string_view>& shortcut,
									   const std::optional<std::string_view>& group, const int sortOrderKey) -> void
{
	menuData_[std::string{ menu }].addItem(group.value_or(""),
										   MenuItemEntryAction{ sortOrderKey, std::string{ label }, action, shortcut });
}

auto ApplicationContext::addMenuToggleAction(bool& toggleValue, const ToggleAction& onToggleChanged,
											 const std::string_view menu, const std::string_view label,
											 const std::optional<std::string_view>& shortcut,
											 const std::optional<std::string_view>& group, const int sortOrderKey)
	-> void
{
	menuData_[std::string{ menu }].addItem(group.value_or(""),
										   MenuItemEntryAction{ sortOrderKey, std::string{ label },
																ToggleEntryAction{ &toggleValue, onToggleChanged },
																shortcut });
}

auto ApplicationContext::addMenuBarTray(const Action& trayDrawCallback) -> void
{
	trayCallbacks_.push_back(trayDrawCallback);
}

auto ApplicationContext::MenuItemEntry::addItem(const std::string_view group, const MenuItemEntryAction& actionEntry)
	-> void
{
	auto& items = groups[std::string{ group }];

	items.push_back(actionEntry);
	std::ranges::sort(items,
			  [](const MenuItemEntryAction& a, const MenuItemEntryAction& b) { return a.sortKey < b.sortKey; });
}
