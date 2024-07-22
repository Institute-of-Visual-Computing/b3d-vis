#pragma once

#include <memory>
#include "FontCollection.h"

class GLFWwindow;
class DebugDrawList;
class GizmoHelper;

class ApplicationContext final
{
public:
	[[nodiscard]] auto getFontCollection() -> FontCollection&
	{
		return fonts_;
	}

	auto setExternalDrawLists(std::shared_ptr<DebugDrawList> debugDrawList, std::shared_ptr<GizmoHelper> gizmoHelper)
		-> void;

	[[nodiscard]] auto getGizmoHelper() const -> std::shared_ptr<GizmoHelper>;
	[[nodiscard]] auto getDrawList() const -> std::shared_ptr<DebugDrawList>;

	GLFWwindow* mainWindowHandle_{};

private:
	FontCollection fonts_{};

	std::shared_ptr<DebugDrawList> debugDrawList_{};
	std::shared_ptr<GizmoHelper> gizmoHelper_{};
};
