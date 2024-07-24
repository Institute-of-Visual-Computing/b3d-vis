#pragma once

#include <memory>
#include <vector>
#include "FontCollection.h"
#include <functional>
#include <string_view>

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
	auto addTool(std::string_view iconLable, Action action) -> void;
	//auto registerAsyncTasks(asyncEngine& )


private:
	FontCollection fonts_{};

	std::shared_ptr<DebugDrawList> debugDrawList_{};
	std::shared_ptr<GizmoHelper> gizmoHelper_{};

	public:
	std::vector<UpdatableComponentBase*> updatableComponents_{};
	std::vector<RendererExtensionBase*> rendererExtensions_{};
	std::unique_ptr<Dockspace> mainDockspace_{nullptr};
};
