#include "NanoViewer.h"

#include "passes/DebugDrawPass.h"
#include "passes/FullscreenTexturePass.h"
#include "passes/InfinitGridPass.h"

#include "InteropUtils.h"

#include <Logging.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <format>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <owl/helper/cuda.h>
#include <print>
#include <stb_image.h>
#include <string>
#include <tracy/Tracy.hpp>

#include <ImGuizmo.h>

#include <IconsFontAwesome6Brands.h>
#include <IconsLucide.h>

#include <boost/process.hpp>

#include "views/ServerConnectSettingsView.h"

#include <string_view>

#include "GizmoOperationFlags.h"



using namespace owl;

namespace
{
	


	auto currentGizmoOperation = GizmoOperationFlags(GizmoOperationFlagBits::none);
	auto currentGizmoMode(ImGuizmo::LOCAL);

	auto keyboardKey(GLFWwindow* window, const unsigned int key) -> void
	{
		auto& io = ImGui::GetIO();
		if (!io.WantCaptureKeyboard)
		{
		}
	}

	auto keyboardSpecialKey(GLFWwindow* window, const int key, [[maybe_unused]] int scancode, const int action,
							const int mods) -> void
	{

		if (action == GLFW_PRESS)
		{
		}
	}

	auto mouseMotion(GLFWwindow* window, const double x, const double y) -> void
	{
		const auto& io = ImGui::GetIO();
		if (!io.WantCaptureMouse)
		{
		}
	}

	auto mouseButton(GLFWwindow* window, const int button, const int action, const int mods) -> void
	{
		const auto& io = ImGui::GetIO();
		if (!io.WantCaptureMouse)
		{
		}
		else
		{
		}
	}

	std::vector<ImFont*> defaultFonts;
	std::unordered_map<float, int> scaleToFont{};
	int currentFontIndex{ 0 };

	auto rebuildFont() -> void
	{
		auto& io = ImGui::GetIO();

		io.Fonts->ClearFonts();
		defaultFonts.clear();

		constexpr auto baseFontSize = 16.0f;

		ImFontConfig config;

		config.OversampleH = 8;
		config.OversampleV = 8;

		auto monitorCount = 0;
		const auto monitors = glfwGetMonitors(&monitorCount);
		assert(monitorCount > 0);
		for (auto i = 0; i < monitorCount; i++)
		{
			const auto monitor = monitors[i];
			auto scaleX = 0.0f;
			auto scaleY = 0.0f;
			glfwGetMonitorContentScale(monitor, &scaleX, &scaleY);
			const auto dpiScale = scaleX; // / 96;

			config.SizePixels = dpiScale * baseFontSize;

			if (!scaleToFont.contains(scaleX))
			{
				auto font =
					io.Fonts->AddFontFromFileTTF("resources/fonts/Roboto-Medium.ttf", dpiScale * baseFontSize, &config);


				static auto iconRangesLucide = ImVector<ImWchar>{};
				ImFontGlyphRangesBuilder builder;
				builder.AddText(ICON_LC_ROTATE_3D ICON_LC_MOVE_3D ICON_LC_SCALE_3D ICON_LC_BAR_CHART_3 ICON_LC_UNPLUG ICON_LC_LOG_OUT ICON_LC_CIRCLE_GAUGE ICON_LC_BUG);
				builder.BuildRanges(&iconRangesLucide);

				const auto iconFontSize = dpiScale * baseFontSize * 2.0f / 3.0f;
				config.MergeMode = true;
				config.PixelSnapH = true;
				config.GlyphMinAdvanceX = iconFontSize;
				config.OversampleH = 8;
				config.OversampleV = 8;

				font = io.Fonts->AddFontFromFileTTF("resources/fonts/lucide.ttf", iconFontSize, &config,
													iconRangesLucide.Data);

				static auto iconRangesFontAwesomeBrands = ImVector<ImWchar>{};
				builder.Clear();
				builder.AddText(ICON_FA_GITHUB);
				builder.BuildRanges(&iconRangesFontAwesomeBrands);

				font = io.Fonts->AddFontFromFileTTF("resources/fonts/fa-brands-400.ttf", iconFontSize,
													&config, iconRangesFontAwesomeBrands.Data);

				config.GlyphMinAdvanceX = iconFontSize * 2.0f;
				config.MergeMode = false;
				auto fontBig = io.Fonts->AddFontFromFileTTF("resources/fonts/lucide.ttf", iconFontSize * 2.0f, &config,
													iconRangesLucide.Data);

				const auto fontIndex = defaultFonts.size();
				defaultFonts.push_back(font);
				defaultFonts.push_back(fontBig);
				scaleToFont[scaleX] = fontIndex;
			}
		}
	}

	auto windowContentScaleCallback([[maybe_unused]] GLFWwindow* window, const float scaleX,
									[[maybe_unused]] float scaleY)
	{
		if (!scaleToFont.contains(scaleX))
		{
			rebuildFont();
		}

		currentFontIndex = scaleToFont[scaleX];
		const auto dpiScale = scaleX; // / 96;
		ImGui::GetStyle().ScaleAllSizes(dpiScale);
	}

	auto onGLFWErrorCallback(int error, const char* description)
	{
		b3d::renderer::log(std::format("Error: {}\n", description), b3d::renderer::LogLevel::error);
	}


	auto initializeGui(GLFWwindow* window) -> void
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		ImGuizmo::AllowAxisFlip(false);
		auto& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls

		ImGui::StyleColorsDark();

		rebuildFont();

		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init();
	}

	auto deinitializeGui() -> void
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	std::unique_ptr<FullscreenTexturePass> fsPass;
	std::unique_ptr<InfinitGridPass> igPass;
	std::unique_ptr<DebugDrawPass> ddPass;

	std::unique_ptr<DebugDrawList> ddList;

	struct ViewerSettings
	{
		float lineWidth{ 4.0 };
		std::array<float, 3> gridColor{ 0.95f, 0.9f, 0.92f };
		bool enableDebugDraw{ true };
		bool enableGridFloor{ true };
	};

	ViewerSettings viewerSettings{};

} // namespace

auto NanoViewer::gui() -> void
{
	static auto showDemoWindow = true;
	ImGui::ShowDemoWindow(&showDemoWindow);

	currentRenderer_->gui();
	static auto showViewerSettings = true;
	ImGui::Begin("Viewer Settings", &showViewerSettings, ImGuiWindowFlags_AlwaysAutoResize);

	const auto& preview = registeredRendererNames_[selectedRendererIndex_];

	if (ImGui::BeginCombo("Renderer", preview.c_str()))
	{
		for (auto n = 0; n < registeredRendererNames_.size(); n++)
		{
			const auto isSelected = (selectedRendererIndex_ == n);
			if (ImGui::Selectable(registeredRendererNames_[n].c_str(), isSelected))
			{
				newSelectedRendererIndex_ = n;
			}

			if (isSelected)
			{
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}


	ImGui::Separator();

	ImGui::Separator();


	ImGui::Checkbox("Enable Grid Floor", &viewerSettings.enableGridFloor);

	if (viewerSettings.enableGridFloor)
	{
		ImGui::SeparatorText("Grid Settings");
		ImGui::ColorEdit3("Color", viewerSettings.gridColor.data());
		ImGui::Separator();
	}

	ImGui::Checkbox("Enable Debug Draw", &viewerSettings.enableDebugDraw);

	if (viewerSettings.enableDebugDraw)
	{
		ImGui::SeparatorText("Debug Draw Settings");
		ImGui::SliderFloat("Line Width", &viewerSettings.lineWidth, 1.0f, 10.0f);
		ImGui::Separator();
		if (ImGui::IsKeyPressed(ImGuiKey_1, false))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::scale);
		}
		if (ImGui::IsKeyPressed(ImGuiKey_2, false))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::translate);
		}
		if (ImGui::IsKeyPressed(ImGuiKey_3, false))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::rotate);
		}
	}

	ImGui::SeparatorText("NVML Settings");


	static auto enablePersistenceMode{ false };
	static auto enabledPersistenceMode{ false };
	static auto showPermissionDeniedMessage{ false };

	uint32_t clock;
	{
		const auto error = nvmlDeviceGetClockInfo(nvmlDevice_, NVML_CLOCK_SM, &clock);
		assert(error == NVML_SUCCESS);
	}

	ImGui::BeginDisabled(!isAdmin_);
	ImGui::Checkbox(std::format("Max GPU SM Clock [current: {} MHz]", clock).c_str(), &enablePersistenceMode);
	ImGui::EndDisabled();
	if (enablePersistenceMode != enabledPersistenceMode)
	{
		if (enablePersistenceMode)
		{
			const auto error =
				nvmlDeviceSetGpuLockedClocks(nvmlDevice_, static_cast<unsigned int>(NVML_CLOCK_LIMIT_ID_TDP),
											 static_cast<unsigned int>(NVML_CLOCK_LIMIT_ID_TDP));

			enabledPersistenceMode = true;

			assert(error == NVML_SUCCESS);
		}

		else
		{
			const auto error = nvmlDeviceResetGpuLockedClocks(nvmlDevice_);

			enabledPersistenceMode = false;
			enablePersistenceMode = false;
			assert(error == NVML_SUCCESS);
		}
	}


	if (!isAdmin_)
	{
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 0.9f, 0.1f, 0.1f, 1.0f });
		ImGui::TextWrapped("This Application should run in admin mode to apply the effect of this option!");
		ImGui::PopStyleColor();
		ImGui::AlignTextToFramePadding();
	}
	// const auto rr = nvmlDeviceSetPersistenceMode(nvmlDevice, enablePersistenceMode?NVML_FEATURE_ENABLED:
	// NVML_FEATURE_DISABLED);


	ImGui::End();
}

auto NanoViewer::render() -> void
{
	constexpr auto layout = static_cast<GLuint>(GL_LAYOUT_GENERAL_EXT);

	const auto cam = b3d::renderer::Camera{ .origin = owl_cast(camera_.getFrom()),
											.at = owl_cast(camera_.getAt()),
											.up = owl_cast(camera_.getUp()),
											.cosFoV = camera_.getCosFovY(),
											.FoV = glm::radians(camera_.getFovYInDegrees()) };

	renderingData_.data.view = b3d::renderer::View{
		.cameras = { cam, cam },
		.mode = b3d::renderer::RenderMode::mono,
	};

	renderingData_.data.renderTargets = b3d::renderer::RenderTargets{
		.colorRt = { viewport3dResources_.cuFramebufferTexture,
					 { static_cast<uint32_t>(viewport3dResources_.framebufferSize.x),
					   static_cast<uint32_t>(viewport3dResources_.framebufferSize.y), 1 } },
		.minMaxRt = { viewport3dResources_.cuFramebufferTexture,
					  { static_cast<uint32_t>(viewport3dResources_.framebufferSize.x),
						static_cast<uint32_t>(viewport3dResources_.framebufferSize.y), 1 } },
	};

	// GL_CALL(glSignalSemaphoreEXT(synchronizationResources_.glSignalSemaphore, 0, nullptr, 0, nullptr, &layout));

	currentRenderer_->render();

	// NOTE: this function call return error, when the semaphore wasn't used before (or it could be in the wrong initial
	// state)
	// GL_CALL(glWaitSemaphoreEXT(synchronizationResources_.glWaitSemaphore, 0, nullptr, 0, nullptr, nullptr));
}
auto NanoViewer::resize(const int width, const int height) -> void
{

	glfwMakeContextCurrent(handle_);
	/*if (framebufferPointer_)
	{
		OWL_CUDA_CHECK(cudaFree(framebufferPointer_));
	}
	OWL_CUDA_CHECK(cudaMallocManaged(&framebufferPointer_, width * height * sizeof(uint32_t)));

	framebufferSize_ = { width, height };
	if (framebufferTexture_ == 0)
	{
		GL_CALL(glGenTextures(1, &framebufferTexture_));
	}
	else
	{
		if (cuFramebufferTexture_)
		{
			OWL_CUDA_CHECK(cudaGraphicsUnregisterResource(cuFramebufferTexture_));
			cuFramebufferTexture_ = 0;
		}
	}

	GL_CALL(glBindTexture(GL_TEXTURE_2D, framebufferTexture_));
	GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));

	OWL_CUDA_CHECK(cudaGraphicsGLRegisterImage(&cuFramebufferTexture_, framebufferTexture_, GL_TEXTURE_2D, 0));*/

	// TODO: make camera change aspect ratio
	// cameraChanged();
}

// auto NanoViewer::cameraChanged() -> void
//{
// }

auto NanoViewer::onFrameBegin() -> void
{
	if (newSelectedRendererIndex_ != selectedRendererIndex_)
	{
		selectRenderer(newSelectedRendererIndex_);
	}
}

NanoViewer::NanoViewer(const std::string& title, const int initWindowWidth, const int initWindowHeight,
					   bool enableVsync, const int rendererIndex)
	: resources_{}, renderingData_{}, colorMapResources_{}
{

	glfwSetErrorCallback(onGLFWErrorCallback);

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_VISIBLE, true);

	handle_ = glfwCreateWindow(initWindowWidth, initWindowHeight, title.c_str(), NULL, NULL);
	if (!handle_)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetWindowUserPointer(handle_, this);
	glfwMakeContextCurrent(handle_);
	glfwSwapInterval((enableVsync) ? 1 : 0);

	debugDrawList_ = std::make_unique<DebugDrawList>();
	gizmoHelper_ = std::make_unique<GizmoHelper>();


	nvmlInit();


	{
		const auto error =
			nvmlDeviceGetHandleByIndex(renderingData_.data.rendererInitializationInfo.deviceIndex, &nvmlDevice_);
		assert(error == NVML_SUCCESS);
	}

	{
		const auto error = nvmlDeviceResetGpuLockedClocks(nvmlDevice_);
		if (error == NVML_ERROR_NO_PERMISSION)
		{
			isAdmin_ = false;
		}
		if (error == NVML_SUCCESS)
		{
			isAdmin_ = true;
		}
		assert(error == NVML_SUCCESS || error == NVML_ERROR_NO_PERMISSION);
	}


	gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
	gladLoadGL();


	// Create Colormap and load data from default colormap, if present
	{
		GL_CALL(glGenTextures(1, &colorMapResources_.colormapTexture));
		GL_CALL(glBindTexture(GL_TEXTURE_2D, colorMapResources_.colormapTexture));

		// Setup filtering parameters for display
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
						GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

		// Load default colormap
		auto filePath = std::filesystem::path{ "resources/colormaps" };
		if (std::filesystem::exists(filePath / "defaultColorMap.json"))
		{
			colorMapResources_.colorMap = b3d::tools::colormap::load(filePath / "defaultColorMap.json");

			if (std::filesystem::path(colorMapResources_.colorMap.colorMapFilePath).is_relative())
			{
				filePath /= colorMapResources_.colorMap.colorMapFilePath;
			}
			else
			{
				filePath = colorMapResources_.colorMap.colorMapFilePath;
			}
			int x, y, n;

			const auto bla = stbi_info(filePath.string().c_str(), &x, &y, &n);

			auto data = stbi_loadf(filePath.string().c_str(), &x, &y, &n, 0);

			// Load Colormap
			GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, x, y, 0, GL_RGBA, GL_FLOAT, data));

			stbi_image_free(data);

			renderingData_.data.colorMapTexture.extent =
				b3d::renderer::Extent{ static_cast<uint32_t>(x), static_cast<uint32_t>(y), 1 };
			renderingData_.data.colorMapTexture.nativeHandle =
				reinterpret_cast<void*>(colorMapResources_.colormapTexture);
		}
		else
		{
			GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 512, 1, 0, GL_RGBA, GL_FLOAT, nullptr));
			renderingData_.data.colorMapTexture.extent = b3d::renderer::Extent{ 512, 1, 1 };
		}

		std::string colormaptexturename = "ColorMap";
		GL_CALL(glObjectLabel(GL_TEXTURE, colorMapResources_.colormapTexture, colormaptexturename.length() + 1,
							  colormaptexturename.c_str()));

		// TODO: add cuda error checks
		auto rc =
			cudaGraphicsGLRegisterImage(&colorMapResources_.cudaGraphicsResource, colorMapResources_.colormapTexture,
										GL_TEXTURE_2D, cudaGraphicsRegisterFlagsTextureGather);

		renderingData_.data.colorMapTexture.target = colorMapResources_.cudaGraphicsResource;

		renderingData_.data.coloringInfo =
			b3d::renderer::ColoringInfo{ b3d::renderer::ColoringMode::single, vec4f{ 1, 1, 1, 1 },
										 colorMapResources_.colorMap.firstColorMapYTextureCoordinate };

		renderingData_.data.colorMapInfos =
			b3d::renderer::ColorMapInfos{ &colorMapResources_.colorMap.colorMapNames,
										  colorMapResources_.colorMap.colorMapCount,
										  colorMapResources_.colorMap.firstColorMapYTextureCoordinate,
										  colorMapResources_.colorMap.colorMapHeightNormalized };
	}

	// Transfer function
	{
		GL_CALL(glGenTextures(1, &transferFunctionResources_.transferFunctionTexture));
		GL_CALL(glBindTexture(GL_TEXTURE_2D, transferFunctionResources_.transferFunctionTexture));

		// Setup filtering parameters for display
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
						GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same


		std::array<float, 512> initBufferData;

		std::ranges::fill(initBufferData, 1);
		GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 512, 1, 0, GL_RED, GL_FLOAT, initBufferData.data()));

		std::string transferFunctionBufferName = "TransferFunctionTexture";
		GL_CALL(glObjectLabel(GL_TEXTURE, transferFunctionResources_.transferFunctionTexture,
							  transferFunctionBufferName.length() + 1, transferFunctionBufferName.c_str()));

		cudaError rc = cudaGraphicsGLRegisterImage(
			&transferFunctionResources_.cudaGraphicsResource, transferFunctionResources_.transferFunctionTexture,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsTextureGather | cudaGraphicsRegisterFlagsWriteDiscard);

		renderingData_.data.transferFunctionTexture.extent = { 512, 1, 1 };
		renderingData_.data.transferFunctionTexture.target = transferFunctionResources_.cudaGraphicsResource;
		renderingData_.data.transferFunctionTexture.nativeHandle =
			reinterpret_cast<void*>(transferFunctionResources_.transferFunctionTexture);
	}

	// NOTE: rendererInfo will be fed into renderer initialization

	selectRenderer(rendererIndex);
	newSelectedRendererIndex_ = selectedRendererIndex_;

	for (auto i = 0; i < b3d::renderer::registry.size(); i++)
	{
		registeredRendererNames_.push_back(b3d::renderer::registry[i].name);
	}
}

auto NanoViewer::showAndRunWithGui() -> void
{
	showAndRunWithGui([&]() { return isRunning_; });
}

auto NanoViewer::drawGizmos(const CameraMatrices& cameraMatrices, const glm::vec2& position, const glm::vec2& size)
	-> void
{
	ImGuizmo::SetDrawlist(); // TODO: set before if statement, oterwise it can lead to crashes
	ImGuizmo::SetRect(position.x, position.y, size.x, size.y);
	if (currentGizmoOperation != GizmoOperationFlagBits::none)
	{

		auto guizmoOperation = ImGuizmo::OPERATION{};
		if (currentGizmoOperation.containsBit(GizmoOperationFlagBits::rotate))
		{
			guizmoOperation = guizmoOperation | ImGuizmo::ROTATE;
		}
		if (currentGizmoOperation.containsBit(GizmoOperationFlagBits::translate))
		{
			guizmoOperation = guizmoOperation | ImGuizmo::TRANSLATE;
		}
		if (currentGizmoOperation.containsBit(GizmoOperationFlagBits::scale))
		{
			guizmoOperation = guizmoOperation | ImGuizmo::SCALE;
		}

		
		for (const auto transform : gizmoHelper_->getTransforms())
		{
			float mat[16];

			mat[3] = 0.0f;
			mat[7] = 0.0f;
			mat[11] = 0.0f;

			mat[12] = transform->p.x;
			mat[13] = transform->p.y;
			mat[14] = transform->p.z;

			mat[15] = 1.0f;

			mat[0] = transform->l.vx.x;
			mat[1] = transform->l.vx.y;
			mat[2] = transform->l.vx.z;

			mat[4] = transform->l.vy.x;
			mat[5] = transform->l.vy.y;
			mat[6] = transform->l.vy.z;

			mat[8] = transform->l.vz.x;
			mat[9] = transform->l.vz.y;
			mat[10] = transform->l.vz.z;
			ImGuizmo::Manipulate(reinterpret_cast<const float*>(&cameraMatrices.view),
								 reinterpret_cast<const float*>(&cameraMatrices.projection), guizmoOperation,
								 currentGizmoMode, mat, nullptr, nullptr);

			transform->p.x = mat[12];
			transform->p.y = mat[13];
			transform->p.z = mat[14];

			transform->l.vx = owl::vec3f{ mat[0], mat[1], mat[2] };
			transform->l.vy = owl::vec3f{ mat[4], mat[5], mat[6] };
			transform->l.vz = owl::vec3f{ mat[8], mat[9], mat[10] };
		}
	}
	auto blockInput = false;


	for (const auto& [bound, transform, worldTransform] : gizmoHelper_->getBoundTransforms())
	{
		float mat[16];

		mat[3] = 0.0f;
		mat[7] = 0.0f;
		mat[11] = 0.0f;

		mat[12] = transform->p.x;
		mat[13] = transform->p.y;
		mat[14] = transform->p.z;

		mat[15] = 1.0f;

		mat[0] = transform->l.vx.x;
		mat[1] = transform->l.vx.y;
		mat[2] = transform->l.vx.z;

		mat[4] = transform->l.vy.x;
		mat[5] = transform->l.vy.y;
		mat[6] = transform->l.vy.z;

		mat[8] = transform->l.vz.x;
		mat[9] = transform->l.vz.y;
		mat[10] = transform->l.vz.z;


		const auto halfSize = bound / 2.0f;

		const auto bounds = std::array{ halfSize.x, halfSize.y, halfSize.z, -halfSize.x, -halfSize.y, -halfSize.z };

		glm::mat4 worldTransformMat{ { worldTransform.l.vx.x, worldTransform.l.vx.y, worldTransform.l.vx.z, 0.0f },
									 { worldTransform.l.vy.x, worldTransform.l.vy.y, worldTransform.l.vy.z, 0.0f },
									 { worldTransform.l.vz.x, worldTransform.l.vz.y, worldTransform.l.vz.z, 0.0f },
									 { worldTransform.p.x, worldTransform.p.y, worldTransform.p.z, 1.0f } };
		const auto matX = cameraMatrices.view * worldTransformMat;

		ImGuizmo::Manipulate(reinterpret_cast<const float*>(&matX),
							 reinterpret_cast<const float*>(&cameraMatrices.projection), ImGuizmo::OPERATION::BOUNDS,
							 currentGizmoMode, mat, nullptr, nullptr, bounds.data());
		if (ImGuizmo::IsUsing())
		{
			blockInput = true;
		}

		transform->p.x = mat[12];
		transform->p.y = mat[13];
		transform->p.z = mat[14];

		transform->l.vx = owl::vec3f{ mat[0], mat[1], mat[2] };
		transform->l.vy = owl::vec3f{ mat[4], mat[5], mat[6] };
		transform->l.vz = owl::vec3f{ mat[8], mat[9], mat[10] };
	}

	if (blockInput)
	{
#if IMGUI_VERSION_NUM >= 18723
		ImGui::SetNextFrameWantCaptureMouse(true);
#else
		ImGui::CaptureMouseFromApp();
#endif
	}
}
auto NanoViewer::computeViewProjectionMatrixFromCamera(const Camera& camera, const int width, const int height)
	-> CameraMatrices
{
	const auto aspect = width / static_cast<float>(height);

	CameraMatrices mat;
	mat.projection = glm::perspective(glm::radians(camera.getFovYInDegrees()), aspect, 0.01f, 10000.0f);
	mat.view = glm::lookAt(camera.getFrom(), camera.getAt(), glm::normalize(camera.getUp()));


	mat.viewProjection = mat.projection * mat.view;
	return mat;
}

auto NanoViewer::initializeViewport3dResources(const int width, const int height) -> void
{
	glfwMakeContextCurrent(handle_);


	if (viewport3dResources_.framebufferPointer)
	{
		OWL_CUDA_CHECK(cudaFree(viewport3dResources_.framebufferPointer));
	}
	OWL_CUDA_CHECK(cudaMallocManaged(&viewport3dResources_.framebufferPointer, width * height * sizeof(uint32_t)));

	viewport3dResources_.framebufferSize = { width, height };
	if (viewport3dResources_.framebufferTexture == 0)
	{
		GL_CALL(glGenTextures(1, &viewport3dResources_.framebufferTexture));
	}
	else
	{
		if (viewport3dResources_.cuFramebufferTexture)
		{
			OWL_CUDA_CHECK(cudaGraphicsUnregisterResource(viewport3dResources_.cuFramebufferTexture));
			viewport3dResources_.cuFramebufferTexture = 0;
		}
	}

	GL_CALL(glBindTexture(GL_TEXTURE_2D, viewport3dResources_.framebufferTexture));
	GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
	GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

	OWL_CUDA_CHECK(cudaGraphicsGLRegisterImage(&viewport3dResources_.cuFramebufferTexture,
											   viewport3dResources_.framebufferTexture, GL_TEXTURE_2D, 0));

	/*GLuint t;
	GL_CALL(glGenTextures(1, &t));*/
	GL_CALL(glGenFramebuffers(1, &viewport3dResources_.framebuffer));

	GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, viewport3dResources_.framebuffer));
	GL_CALL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
								   viewport3dResources_.framebufferTexture, 0));
	GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

auto cleanupViewport3DResources() -> void
{
}


auto NanoViewer::updateViewport3dResources() -> void
{
}

auto NanoViewer::renderViewport3d(const int width, const int height) -> void
{
	if (viewport3dResources_.framebufferSize.x != width || viewport3dResources_.framebufferSize.y != height)
	{
		initializeViewport3dResources(width, height);
	}
	const auto cameraMatrices = computeViewProjectionMatrixFromCamera(camera_, width, height);

	GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, viewport3dResources_.framebuffer));
	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	render();

	fsPass->setViewport(width, height);
	fsPass->setSourceTexture(viewport3dResources_.framebufferTexture);
	fsPass->execute();


	if (viewerSettings.enableGridFloor)
	{
		igPass->setViewProjectionMatrix(cameraMatrices.viewProjection);
		igPass->setViewport(width, height);
		igPass->setGridColor(
			glm::vec3{ viewerSettings.gridColor[0], viewerSettings.gridColor[1], viewerSettings.gridColor[2] });
		igPass->execute();
	}

	if (viewerSettings.enableDebugDraw)
	{
		ddPass->setViewProjectionMatrix(cameraMatrices.viewProjection);
		ddPass->setViewport(width, height);
		ddPass->setLineWidth(viewerSettings.lineWidth);
		ddPass->execute();
	}

	GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}


auto NanoViewer::draw() -> void
{
	ZoneScoped;

	// TODO: if windows minimized or not visible -> skip rendering
	onFrameBegin();
	glClear(GL_COLOR_BUFFER_BIT);
	static double lastCameraUpdate = -1.f;
	
	gizmoHelper_->clear();

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGui::PushFont(defaultFonts[currentFontIndex]);
	//TODO: Investigate if this combination is alwys intercepted by OS
	if(ImGui::IsKeyDown(ImGuiMod_Alt) and ImGui::IsKeyPressed(ImGuiKey_F4, false))
	{
		isRunning_ = false;
	}

	static auto connectView = ServerConnectSettingsView{ "Server Connect", []() { std::println("submit!!!"); } };

	ImGui::BeginMainMenuBar();
	if (ImGui::BeginMenu("Program"))
	{
		if (ImGui::MenuItem(ICON_LC_UNPLUG " Data Service..", nullptr, nullptr))
		{

		}
		if (ImGui::MenuItem("Server Connection...", nullptr, nullptr))
		{
			connectView.open();
			connectView.reset();
		}

		if (ImGui::MenuItem(ICON_LC_LOG_OUT " Quit", "Alt+F4", nullptr))
		{
			isRunning_ = false;
		}

		ImGui::EndMenu();
	}


	if (ImGui::BeginMenu("Tools"))
	{
		if (ImGui::MenuItem(ICON_LC_BAR_CHART_3 " Histogram", nullptr, nullptr))
		{
		}

		if (ImGui::MenuItem("Transfer Function", nullptr, nullptr))
		{
		}

		ImGui::EndMenu();
	}

	if (ImGui::BeginMenu("Help"))
	{
		const auto url = "https://github.com/Institut-of-Visual-Computing/b3d-vis";

		if (ImGui::MenuItem(ICON_FA_GITHUB " Source Code", nullptr, nullptr))
		{
			auto cmd = "";
#ifdef __APPLE__
#ifdef TARGET_OS_MAC
			cmd = "open";
#endif
#elif __linux__
			cmd = "xdg-open";
#elif _WIN32
			cmd = "start";
#else

#endif
			std::system(std::format("{} {}", cmd, url).c_str());
		}
		ImGui::SeparatorText("Develop Tools");
		ImGui::MenuItem(ICON_LC_BUG" Debug Options");
		ImGui::MenuItem(ICON_LC_CIRCLE_GAUGE" Renderer Profiler");

		ImGui::Separator();
		ImGui::MenuItem("About", nullptr, nullptr);
		ImGui::EndMenu();
	}

	ImGui::EndMainMenuBar();

	

	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->WorkPos);
	ImGui::SetNextWindowSize(viewport->WorkSize);
	ImGui::Begin("Editor", 0,
				 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus);


	static ImGuiWindowClass windowClass;
	static ImGuiID dockspaceId = 0;

	dockspaceId = ImGui::GetID("mainDock");


	ImGui::DockSpace(dockspaceId);

	windowClass.ClassId = dockspaceId;
	windowClass.DockingAllowUnclassed = true;
	// windowClass.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_AutoHideTabBar;
	//  ImGuiDockNodeFlags_NoWindowMenuButton;


	ImGui::End();

	/*ImGui::SetNextWindowClass(&windowClass);
	ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);*/


	ImGui::SetNextWindowClass(&windowClass);
	ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);

	gui();
	windowClass.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_AutoHideTabBar | ImGuiDockNodeFlags_NoUndocking;
	ImGui::SetNextWindowClass(&windowClass);
	ImGui::Begin("VolumeViewport", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse);

	auto viewport3dSize = ImGui::GetContentRegionAvail();
	ImVec2 p = ImGui::GetCursorScreenPos();
	ImGui::SetNextItemAllowOverlap();
	ImGui::InvisibleButton("##volumeViewport", viewport3dSize,
						   ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);


	static auto moveCameraFaster = false;
	auto& io = ImGui::GetIO();


	if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
	{
		moveCameraFaster = true;
	}

	if (ImGui::IsKeyReleased(ImGuiKey_LeftShift))
	{
		moveCameraFaster = false;
	}

	const auto fastSpeed = 25.0f;
	const auto cameraMoveVelocity = 0.0f;
	auto cameraMoveAcceleration = glm::vec3{ 0 };
	const auto maxCameraMoveAcceleration = 1.0f;
	static auto AccelerationExpire = 0.0;
	const auto sensitivity = 0.1f;
	if (ImGui::IsKeyDown(ImGuiKey_W))
	{
		cameraMoveAcceleration = camera_.forward_ * camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f);
	}
	if (ImGui::IsKeyDown(ImGuiKey_S))
	{
		cameraMoveAcceleration =
			-camera_.forward_ * camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f);
	}
	if (ImGui::IsKeyDown(ImGuiKey_A))
	{
		cameraMoveAcceleration = -glm::normalize(glm::cross(camera_.forward_, camera_.getUp())) *
			camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f);
	}
	if (ImGui::IsKeyDown(ImGuiKey_D))
	{
		cameraMoveAcceleration = glm::normalize(glm::cross(camera_.forward_, camera_.getUp())) *
			camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f);
	}

	auto delta = io.MouseDelta;
	delta.x *= -1.0;


	if (ImGui::IsItemActive())
	{
		if (!ImGuizmo::IsUsing())
		{
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
			{
				const auto right = glm::normalize(glm::cross(camera_.forward_, camera_.getUp()));
				cameraMoveAcceleration += -glm::normalize(glm::cross(camera_.forward_, right)) *
					camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f) * delta.y;

				cameraMoveAcceleration +=
					right * camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f) * delta.x;
			}
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
			{

				auto mouseScreenLocation = io.MousePos;
				const auto up = camera_.getUp();
				const auto right = glm::normalize(glm::cross(up, camera_.forward_));

				const auto f = glm::normalize(camera_.forward_ + right * delta.y + up * delta.x);

				auto rotationAxis = glm::normalize(glm::cross(f, camera_.forward_));

				if (glm::length(rotationAxis) >= 0.001f)
				{
					const auto rotation =
						glm::rotate(glm::identity<glm::mat4>(),
									glm::radians(glm::length(glm::vec2{ delta.x, delta.y }) * sensitivity), f);
					camera_.forward_ = glm::normalize(glm::vec3(rotation * glm::vec4(camera_.forward_, 0.0f)));
					camera_.right_ = glm::normalize(glm::cross(camera_.forward_, up));
				}
			}
		}
	}

	camera_.position_ += cameraMoveAcceleration * io.DeltaTime;


	ImGui::SetCursorScreenPos(p);
	ImGui::SetNextItemAllowOverlap();
	ImGui::Image((ImTextureID)viewport3dResources_.framebufferTexture, viewport3dSize, { 0.0f, 1.0f }, { 1.0f, 0.0f });

	if (viewerSettings.enableDebugDraw)
	{
		ImGui::SetNextItemAllowOverlap();
		ImGui::SetCursorScreenPos(p);
		const auto cameraMatrices = computeViewProjectionMatrixFromCamera(camera_, viewport3dSize.x, viewport3dSize.y);
		drawGizmos(cameraMatrices, glm::vec2{ p.x, p.y }, glm::vec2{ viewport3dSize.x, viewport3dSize.y });
	}

	const auto showControls = true;
	if (showControls)
	{
		ImGui::SetNextItemAllowOverlap();

		const auto scale = ImGui::GetWindowDpiScale();
		auto buttonPosition = p + ImVec2(scale * 20, scale * 20);
		ImGui::SetCursorScreenPos(buttonPosition);
		const auto buttonPadding = scale * 4.0f;
		const auto buttonSize = scale * 40;

		const auto activeColor = ImGui::GetStyle().Colors[ImGuiCol_ButtonActive];

		const auto prevOperationState = currentGizmoOperation;

		if (prevOperationState.containsBit(GizmoOperationFlagBits::scale))
		{
			ImGui::PushStyleColor(ImGuiCol_Button, activeColor);
		}

		ImGui::PushFont(defaultFonts[currentFontIndex + 1]);
		if (ImGui::Button(ICON_LC_SCALE_3D "##scale_control_handle", ImVec2{ buttonSize, buttonSize }))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::scale);
		}
		ImGui::PopFont();

		if (ImGui::BeginItemTooltip())
		{
			ImGui::Text("Scale Volume");
			ImGui::TextDisabled("Hotkey: 1");
			ImGui::EndTooltip();
		}

		if (prevOperationState.containsBit(GizmoOperationFlagBits::scale))
		{
			ImGui::PopStyleColor();
		}
		ImGui::SetNextItemAllowOverlap();
		buttonPosition += ImVec2(0, buttonPadding + buttonSize);
		ImGui::SetCursorScreenPos(buttonPosition);

		if (prevOperationState.containsBit(GizmoOperationFlagBits::translate))
		{
			ImGui::PushStyleColor(ImGuiCol_Button, activeColor);
		}
		ImGui::PushFont(defaultFonts[currentFontIndex + 1]);
		if (ImGui::Button(ICON_LC_MOVE_3D "##translate_control_handle", ImVec2{ buttonSize, buttonSize }))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::translate);
		}
		ImGui::PopFont();
		if (ImGui::BeginItemTooltip())
		{
			ImGui::Text("Translate Volume");
			ImGui::TextDisabled("Hotkey: 2");
			ImGui::EndTooltip();
		}

		if (prevOperationState.containsBit(GizmoOperationFlagBits::translate))
		{
			ImGui::PopStyleColor();
		}
		ImGui::SetNextItemAllowOverlap();
		buttonPosition += ImVec2(0, buttonPadding + buttonSize);
		ImGui::SetCursorScreenPos(buttonPosition);

		if (prevOperationState.containsBit(GizmoOperationFlagBits::rotate))
		{
			ImGui::PushStyleColor(ImGuiCol_Button, activeColor);
		}
		ImGui::PushFont(defaultFonts[currentFontIndex + 1]);
		if (ImGui::Button(ICON_LC_ROTATE_3D "##rotate_control_handle", ImVec2{ buttonSize, buttonSize }))
		{
			currentGizmoOperation.flip(GizmoOperationFlagBits::rotate);
		}
		ImGui::PopFont();
		if (ImGui::BeginItemTooltip())
		{
			ImGui::Text("Rotate Volume");
			ImGui::TextDisabled("Hotkey: 3");
			ImGui::EndTooltip();
		}

		if (prevOperationState.containsBit(GizmoOperationFlagBits::rotate))
		{
			ImGui::PopStyleColor();
		}


	}

	ImGui::End();



	
	connectView.draw();


	ImGui::PopFont();
	ImGui::EndFrame();

	ImGui::Render();

	if (viewport3dSize.x > 0 && viewport3dSize.y > 0)
	{
		renderViewport3d(viewport3dSize.x, viewport3dSize.y);
	}

	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		GLFWwindow* backup_current_context = glfwGetCurrentContext();
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
		glfwMakeContextCurrent(backup_current_context);
	}

	glfwSwapBuffers(handle_);
	glfwPollEvents();
	FrameMark;
}

auto NanoViewer::showAndRunWithGui(const std::function<bool()>& keepgoing) -> void
{
	gladLoadGL();

	ddList = std::make_unique<DebugDrawList>();
	fsPass = std::make_unique<FullscreenTexturePass>();
	igPass = std::make_unique<InfinitGridPass>();
	ddPass = std::make_unique<DebugDrawPass>(debugDrawList_.get());

	int width, height;
	glfwGetFramebufferSize(handle_, &width, &height);
	resize(width, height);

	glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
	glfwSetFramebufferSizeCallback(handle_,
								   [](GLFWwindow* window, int width, int height)
								   {
									   auto* viewer = static_cast<NanoViewer*>(glfwGetWindowUserPointer(window));
									   viewer->resize(width, height);
									   viewer->draw();
								   });
	glfwSetMouseButtonCallback(handle_, ::mouseButton);
	glfwSetKeyCallback(handle_, keyboardSpecialKey);
	glfwSetCharCallback(handle_, keyboardKey);
	glfwSetCursorPosCallback(handle_, ::mouseMotion);
	glfwSetWindowContentScaleCallback(handle_, windowContentScaleCallback);

	initializeGui(handle_);
	glfwMakeContextCurrent(handle_);

	while (!glfwWindowShouldClose(handle_) && keepgoing())
	{
		{
			draw();
		}
	}

	deinitializeGui();
	currentRenderer_->deinitialize();
	glfwDestroyWindow(handle_);
	glfwTerminate();
}
NanoViewer::~NanoViewer()
{
	cudaGraphicsUnregisterResource(transferFunctionResources_.cudaGraphicsResource);
	cudaGraphicsUnregisterResource(colorMapResources_.cudaGraphicsResource);

	if (isAdmin_)
	{
		const auto error = nvmlDeviceResetGpuLockedClocks(nvmlDevice_);
		assert(error == NVML_SUCCESS);
	}
	nvmlShutdown();
}
auto NanoViewer::selectRenderer(const std::uint32_t index) -> void
{
	assert(index < b3d::renderer::registry.size());
	if (selectedRendererIndex_ == index)
	{
		return;
	}
	if (currentRenderer_)
	{
		currentRenderer_->deinitialize();
	}


	selectedRendererIndex_ = index;
	currentRenderer_ = b3d::renderer::registry[selectedRendererIndex_].rendererInstance;

	const auto debugInfo = b3d::renderer::DebugInitializationInfo{ debugDrawList_, gizmoHelper_ };

	currentRenderer_->initialize(&renderingData_.buffer, debugInfo);
}
