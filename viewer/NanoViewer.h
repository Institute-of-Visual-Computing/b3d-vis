#pragma once

#include <RendererBase.h>

#include <nvml.h>

#include "Camera.h"
#include "DebugDrawList.h"
#include "GizmoHelper.h"

#include <ProfilersWindow.h>

#include "features/projectExplorer/ProjectExplorer.h"
#include "features/sofiaSearch/SoFiaSearch.h"
#include "features/transferMapping/TransferMapping.h"
#include "framework/ApplicationContext.h"
#include "framework/MenuBar.h"
#include "views/VolumeView.h"

class NanoViewer final
{
public:
	explicit NanoViewer(const std::string& title = "Nano Viewer", int initWindowWidth = 1980,
						int initWindowHeight = 1080, bool enableVsync = false, int rendererIndex = 0);
	auto showAndRunWithGui() -> void;
	auto showAndRunWithGui(const std::function<bool()>& keepgoing) -> void;

	auto enableDevelopmentMode(const bool enable = true) const -> void
	{
		applicationContext_->isDevelopmentModeEnabled = enable;
	}

	[[nodiscard]] auto getCamera() -> ::Camera&
	{
		return camera_;
	}

	[[nodiscard]] auto getApplicationContext() const -> ApplicationContext&
	{
		return *applicationContext_.get();
	}
	~NanoViewer();

private:
	auto selectRenderer(uint32_t index) -> void;
	auto gui() -> void;
	auto draw() -> void;
	auto onFrameBegin() -> void;


	struct CameraMatrices
	{
		glm::mat4 view;
		glm::mat4 projection;
		glm::mat4 viewProjection;
	};

	std::shared_ptr<DebugDrawList> debugDrawList_{};
	std::shared_ptr<GizmoHelper> gizmoHelper_{};

	ProfilersWindow profilersWindow_{};

	std::shared_ptr<b3d::renderer::RendererBase> currentRenderer_{ nullptr };
	std::int32_t selectedRendererIndex_{ -1 };
	std::int32_t newSelectedRendererIndex_{ -1 };
	std::vector<std::string> registeredRendererNames_{};

	nvmlDevice_t nvmlDevice_{};
	bool isAdmin_{ false };

	::Camera camera_{};

	bool isRunning_{ true };


	struct ViewerSettings
	{
		float lineWidth{ 4.0 };
		std::array<float, 3> gridColor{ 0.95f, 0.9f, 0.92f };
		bool enableDebugDraw{ true };
		bool enableGridFloor{ true };
	};

	ViewerSettings viewerSettings_{};

	std::unique_ptr<ApplicationContext> applicationContext_{};
	std::unique_ptr<VolumeView> volumeView_{};
	std::unique_ptr<TransferMapping> transferMapping_{};
	std::unique_ptr<ProjectExplorer> projectExplorer_{};
	std::unique_ptr<SoFiaSearch> soFiaSearch_{};
	std::unique_ptr<MenuBar> mainMenu_{};
	b3d::renderer::RenderingDataWrapper renderingData_{};

	bool showProfiler_{ false };
	bool showDebugOptions_{ false };
	bool showAboutWindow_{ false };
	bool showImGuiDemo_{ false };
};
