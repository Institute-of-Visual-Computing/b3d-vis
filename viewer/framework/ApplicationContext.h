#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "FontCollection.h"

#include "ApplicationSettings.h"
#include "ServerClient.h"
#include "framework/Dockspace.h"

class GLFWwindow;
class DebugDrawList;
class GizmoHelper;
class UpdatableComponentBase;
class RendererExtensionBase;

namespace std_help
{
	template <class... Ts>
	// ReSharper disable once CppInconsistentNaming
	struct overloaded : Ts...
	{
		using Ts::operator()...;
	};
} // namespace std_help
using Action = std::function<void(void)>;
using ToggleAction = std::function<void(bool)>;

class ApplicationContext final
{
public:
	ApplicationContext();

	[[nodiscard]] auto getFontCollection() -> FontCollection&
	{
		return fonts_;
	}

	auto setExternalDrawLists(const std::shared_ptr<DebugDrawList>& debugDrawList,
							  const std::shared_ptr<GizmoHelper>& gizmoHelper)
		-> void;

	[[nodiscard]] auto getGizmoHelper() const -> std::shared_ptr<GizmoHelper>;
	[[nodiscard]] auto getDrawList() const -> std::shared_ptr<DebugDrawList>;

	GLFWwindow* mainWindowHandle_{};

	[[nodiscard]] auto getMainDockspace() const -> Dockspace*;

	auto addUpdatableComponent(UpdatableComponentBase* component) -> void;
	auto removeUpdatableComponent(UpdatableComponentBase* component) -> void;

	auto addRendererExtensionComponent(RendererExtensionBase* component) -> void;
	auto addMenuAction(Action action, std::string_view menu, std::string_view label,
					   const std::optional<std::string_view>& shortcut = std::nullopt,
					   const std::optional<std::string_view>& group = std::nullopt, int sortOrderKey = 0) -> void;
	auto addMenuToggleAction(bool& toggleValue, const ToggleAction& onToggleChanged, std::string_view menu,
							 std::string_view label, const std::optional<std::string_view>& shortcut = std::nullopt,
							 const std::optional<std::string_view>& group = std::nullopt, int sortOrderKey = 0) -> void;
	auto addMenuBarTray(const Action& trayDrawCallback = []() {}) -> void;
	//TODO: investigate if thous API is needed
	//auto addTool(std::string_view iconLabel, Action action) -> void;
	// auto registerAsyncTasks(asyncEngine& )


	struct ToggleEntryAction
	{
		bool* value;
		ToggleAction onChangeAction;
	};

	using ActionHolder = std::variant<ToggleEntryAction, Action>;

	struct MenuItemEntryAction
	{
		int sortKey{ 0 };
		std::string label{};
		ActionHolder action;
		std::optional<std::string_view> shortcut{};
	};

	struct MenuItemEntry
	{
		std::map<std::string, std::vector<MenuItemEntryAction>> groups;

		auto addItem(std::string_view group, const MenuItemEntryAction& actionEntry) -> void;
	};

	std::map<std::string, MenuItemEntry> menuData_;
	std::vector<Action> trayCallbacks_;

	ApplicationSettings settings_{};

private:
	FontCollection fonts_{};

	std::shared_ptr<DebugDrawList> debugDrawList_{};
	std::shared_ptr<GizmoHelper> gizmoHelper_{};

public:
	std::vector<UpdatableComponentBase*> updatableComponents_{};
	std::vector<RendererExtensionBase*> rendererExtensions_{};
	std::unique_ptr<Dockspace> mainDockspace_{ nullptr };

	b3d::tools::project::ServerClient serverClient { };
};
