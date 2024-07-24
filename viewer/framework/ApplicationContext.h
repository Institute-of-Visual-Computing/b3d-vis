#pragma once

#include <functional>
#include <memory>
#include <string_view>
#include <variant>
#include <vector>
#include <map>
#include <optional>

#include "FontCollection.h"

#include "framework/Dockspace.h"

class GLFWwindow;
class DebugDrawList;
class GizmoHelper;
class UpdatableComponentBase;
class RendererExtensionBase;

using Action = std::function<void(void)>;

class ApplicationContext final
{
public:
	ApplicationContext();

	[[nodiscard]] auto getFontCollection() -> FontCollection&
	{
		return fonts_;
	}

	auto setExternalDrawLists(std::shared_ptr<DebugDrawList> debugDrawList, std::shared_ptr<GizmoHelper> gizmoHelper)
		-> void;

	[[nodiscard]] auto getGizmoHelper() const -> std::shared_ptr<GizmoHelper>;
	[[nodiscard]] auto getDrawList() const -> std::shared_ptr<DebugDrawList>;

	GLFWwindow* mainWindowHandle_{};

	[[nodiscard]] auto getMainDockspace() -> Dockspace*;

	auto addUpdatableComponent(UpdatableComponentBase* component) -> void;
	auto addRendererExtensionComponent(RendererExtensionBase* component) -> void;
	auto addMenuAction(std::vector<std::string_view> menuPath, Action action) -> void;
	auto addMenuToggle(std::vector<std::string_view> menuPath, bool& toogle) -> void;
	auto addMenuAction(Action action, std::string_view menu, std::string_view label,
					   std::optional<std::string_view> group = std::nullopt, int sortOrderKey = 0) -> void;
	auto addTool(std::string_view iconLable, Action action) -> void;
	// auto registerAsyncTasks(asyncEngine& )


	struct MenuItemEntryAction
	{
		int sortKey{ 0 };
		std::string label{};
	};

	struct MenuItemEntry
	{
		std::map<std::string, std::vector<MenuItemEntryAction>> groups;

		auto addItem(std::string_view group, MenuItemEntryAction actionEntry) -> void;
	};

	std::map<std::string, MenuItemEntry> menuData;

private:
	FontCollection fonts_{};

	std::shared_ptr<DebugDrawList> debugDrawList_{};
	std::shared_ptr<GizmoHelper> gizmoHelper_{};

public:
	std::vector<UpdatableComponentBase*> updatableComponents_{};
	std::vector<RendererExtensionBase*> rendererExtensions_{};
	std::unique_ptr<Dockspace> mainDockspace_{ nullptr };
};
