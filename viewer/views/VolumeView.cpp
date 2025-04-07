#define GLM_ENABLE_EXPERIMENTAL
#include "VolumeView.h"

#include <print>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <GLFW/glfw3.h>
#include <IconsLucide.h>
#include <ImGuizmo.h>


#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <owl/helper/cuda.h>

#include <RendererBase.h>

#include "GLGpuTimers.h"
#include "GLUtils.h"
#include "GizmoHelper.h"
#include "InteropUtils.h"
#include "framework/ApplicationContext.h"
#include "passes/DebugDrawPass.h"
#include "passes/FullscreenTexturePass.h"
#include "passes/InfinitGridPass.h"

#include <ImGuiProfilerRenderer.h>


namespace
{
	auto computeViewProjectionMatrixFromCamera(const Camera& camera, const int width, const int height)
		-> VolumeView::CameraMatrices
	{
		const auto aspect = width / static_cast<float>(height);

		VolumeView::CameraMatrices mat;
		mat.projection = glm::perspective(glm::radians(camera.getFovYInDegrees()), aspect, 0.01f, 10000.0f);
		mat.view = glm::lookAt(camera.getFrom(), camera.getAt(), glm::normalize(camera.getUp()));
		mat.viewProjection = mat.projection * mat.view;
		return mat;
	}
} // namespace

VolumeView::VolumeView(ApplicationContext& appContext, Dockspace* dockspace)
	: DockableWindowViewBase(appContext, "Volume Viewport", dockspace,
							 /*WindowFlagBits::noTitleBar | WindowFlagBits::noUndocking |*/ WindowFlagBits::hideTabBar |
								 WindowFlagBits::noCollapse)
{
	fullscreenTexturePass_ = std::make_unique<FullscreenTexturePass>();
	infiniteGridPass_ = std::make_unique<InfiniteGridPass>();
	debugDrawPass_ = std::make_unique<DebugDrawPass>(applicationContext_->getDrawList().get());

	camera_.setOrientation(glm::vec3(1.0, 1.0, 1.0), glm::vec3(0.0, 0.0, 0.0), camera_.getUp(),
						   camera_.getFovYInDegrees());

	cameraLastFrame_ = camera_;

	orbitController_.setCamera(&camera_);
	fpsController_.setCamera(&camera_);

	flyAnimationSettings_.radius = 3.0f;
	animator_.addPropertyAnimation(
		[&](const float t, const float dt)
		{
			const auto radius = flyAnimationSettings_.radius;
			const auto height = flyAnimationSettings_.height;
			const auto& origin = flyAnimationSettings_.origin;
			const auto sin = glm::sin(t);
			const auto cos = glm::cos(t);

			const auto offset = glm::vec3{ sin * radius, height, cos * radius };

			const auto lastPosition = cameraLastFrame_.position_;
			const auto position = origin + offset;
			const auto targetVelocity = position - lastPosition;

			const auto cameraCurrentPosition = camera_.position_;
			{

				const auto cameraVelocity = position - camera_.position_;

				auto x = camera_.position_.x;
				auto y = camera_.position_.y;
				auto z = camera_.position_.z;

				auto vx = cameraVelocity.x;
				auto vy = cameraVelocity.y;
				auto vz = cameraVelocity.z;


				animation::springDamperExact2(x, vx, position.x, targetVelocity.x, flyAnimationSettings_.stiffness,
											  flyAnimationSettings_.dumping, dt);
				animation::springDamperExact2(y, vy, position.y, targetVelocity.y, flyAnimationSettings_.stiffness,
											  flyAnimationSettings_.dumping, dt);
				animation::springDamperExact2(z, vz, position.z, targetVelocity.z, flyAnimationSettings_.stiffness,
											  flyAnimationSettings_.dumping, dt);
				camera_.position_ = glm::vec3{ x, y, z };
			}

			{
				auto targetNormalized = glm::normalize(flyAnimationSettings_.target - cameraCurrentPosition);

				auto x = camera_.forward_.x;
				auto y = camera_.forward_.y;
				auto z = camera_.forward_.z;

				const auto forwardVelocity =
					cameraLastFrame_.position_ + cameraLastFrame_.forward_ - (camera_.forward_ + cameraCurrentPosition);


				auto vx = forwardVelocity.x;
				auto vy = forwardVelocity.y;
				auto vz = forwardVelocity.z;
				constexpr auto targetForwardVelocity = glm::vec3{ 0.0, 0.0, 0.0f }; // forwardVelocity;

				animation::springDamperExact2(x, vx, targetNormalized.x, targetForwardVelocity.x,
											  flyAnimationSettings_.stiffness, flyAnimationSettings_.dumping, dt);
				animation::springDamperExact2(y, vy, targetNormalized.y, targetForwardVelocity.y,
											  flyAnimationSettings_.stiffness, flyAnimationSettings_.dumping, dt);
				animation::springDamperExact2(z, vz, targetNormalized.z, targetForwardVelocity.z,
											  flyAnimationSettings_.stiffness, flyAnimationSettings_.dumping, dt);


				camera_.setOrientation(camera_.position_,
									   camera_.position_ + camera_.forward_ + glm::vec3{ vx, vy, vz }, camera_.up_, 60);
			}
		});

	applicationContext_->addMenuToggleAction(
		demoModeEnabled_, [&](const bool isOn) { demoMode(isOn); }, "Help", "Demo Mode", "F2", std::nullopt, 0);
}

VolumeView::~VolumeView()
{
	deinitializeGraphicsResources();
}


auto VolumeView::onDraw() -> void
{
	cameraLastFrame_ = camera_;
	renderVolume();

	if (ImGui::IsKeyPressed(ImGuiKey_1, false))
	{
		currentGizmoOperation_.flip(GizmoOperationFlagBits::scale);
	}
	if (ImGui::IsKeyPressed(ImGuiKey_2, false))
	{
		currentGizmoOperation_.flip(GizmoOperationFlagBits::translate);
	}
	if (ImGui::IsKeyPressed(ImGuiKey_3, false))
	{
		currentGizmoOperation_.flip(GizmoOperationFlagBits::rotate);
	}

	const auto p = ImGui::GetCursorScreenPos();
	ImGui::SetNextItemAllowOverlap();
	ImGui::InvisibleButton("##volumeViewport", viewportSize_,
						   ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);


	// camera control
	getCameraController(cameraControllerType_)->update();

	ImGui::SetCursorScreenPos(p);
	ImGui::SetNextItemAllowOverlap();
	ImGui::Image((ImTextureID)viewGraphicsResources_.viewFbo.colorAttachment, viewportSize_, { 0.0f, 1.0f },
				 { 1.0f, 0.0f }, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });

	if (viewerSettings_.enableDebugDraw)
	{
		ImGui::SetNextItemAllowOverlap();
		ImGui::SetCursorScreenPos(p);
		const auto cameraMatrices = computeViewProjectionMatrixFromCamera(camera_, static_cast<int>(viewportSize_.x),
																		  static_cast<int>(viewportSize_.y));
		drawGizmos(cameraMatrices, glm::vec2{ p.x, p.y }, glm::vec2{ viewportSize_.x, viewportSize_.y });
	}


	if (viewerSettings_.enableControlToolBar)
	{
		ImGui::SetNextItemAllowOverlap();

		const auto scale = ImGui::GetWindowDpiScale();


		auto buttonPosition = p + ImVec2(scale * 20, scale * 20);
		ImGui::SetCursorScreenPos(buttonPosition);
		const auto buttonPadding = scale * 4.0f;
		const auto buttonSize = scale * 40;

		const auto activeColor = ImGui::GetStyle().Colors[ImGuiCol_ButtonActive];

		const auto prevOperationState = currentGizmoOperation_;

		if (prevOperationState.containsBit(GizmoOperationFlagBits::scale))
		{
			ImGui::PushStyleColor(ImGuiCol_Button, activeColor);
		}

		ImGui::PushFont(applicationContext_->getFontCollection().getBigIconsFont());
		if (ImGui::Button(ICON_LC_SCALE_3D "##scale_control_handle", ImVec2{ buttonSize, buttonSize }))
		{
			currentGizmoOperation_.flip(GizmoOperationFlagBits::scale);
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
		ImGui::PushFont(applicationContext_->getFontCollection().getBigIconsFont());
		if (ImGui::Button(ICON_LC_MOVE_3D "##translate_control_handle", ImVec2{ buttonSize, buttonSize }))
		{
			currentGizmoOperation_.flip(GizmoOperationFlagBits::translate);
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
		ImGui::PushFont(applicationContext_->getFontCollection().getBigIconsFont());
		if (ImGui::Button(ICON_LC_ROTATE_3D "##rotate_control_handle", ImVec2{ buttonSize, buttonSize }))
		{
			currentGizmoOperation_.flip(GizmoOperationFlagBits::rotate);
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

		ImGui::SetNextItemAllowOverlap();
		buttonPosition += ImVec2(0, buttonPadding + buttonSize);
		ImGui::SetCursorScreenPos(buttonPosition);

		// camera switch
		{
			static constexpr auto types = std::array{ "orbit", "fly" };
			auto cameraType = static_cast<int>(cameraControllerType_);
			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 12.0f);
			ImGui::SetNextItemWidth(40 * scale);
			ImGui::SliderInt("##cameraType", &cameraType, 0, static_cast<int>(types.size() - 1), types[cameraType]);
			if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
			{
				cameraControllerType_ =
					static_cast<CameraControllerType>((static_cast<int>(cameraControllerType_) + 1) % 2);
			}
			ImGui::PopStyleVar(2);
		}
	}

	if (ImGui::IsKeyPressed(ImGuiKey_F2, false))
	{
		demoMode(!animator_.isRunning());
	}

	if (viewerSettings_.enableFrameGraph)
	{
		const auto cursorPositionY = ImGui::GetCursorPosY();

		const auto remainingSizeY = this->viewportSize_.y - cursorPositionY;
		constexpr auto profilerHeight = 200.0f;

		if (remainingSizeY > profilerHeight)
		{
			const auto scale = ImGui::GetWindowDpiScale();
			const auto controlPadding = scale * 16.0f;
			const auto offsetY = remainingSizeY - profilerHeight - controlPadding;

			const auto windowPosition = ImGui::GetWindowPos();
			ImGui::SetCursorPosY(cursorPositionY + offsetY);
			ImGui::SetCursorPosX(controlPadding);
			{
				const auto canvasSize = ImGui::GetContentRegionAvail();

				const auto sizeMargin = static_cast<int>(ImGui::GetStyle().ItemSpacing.y);
				constexpr auto maxGraphHeight = 140;
				const auto availableGraphHeight = (static_cast<int>(canvasSize.y) - sizeMargin);
				const auto graphHeight = std::min(maxGraphHeight, availableGraphHeight);
				constexpr auto legendWidth = 400;
				const auto graphWidth = static_cast<int>(glm::min(canvasSize.x, 1200.0f)) - legendWidth;
				constexpr auto frameOffset = 0;

				const auto position = ImGui::GetCursorPos();
				ImGui::PushFont(applicationContext_->getFontCollection().getGpuCpuExtraBigTextFont());
				ImGui::SetNextItemAllowOverlap();
				ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 0.6f, 0.6f, 0.6f, 0.4f });
				ImGui::Text("GPU");
				ImGui::PopStyleColor();
				ImGui::PopFont();
				ImGui::SetCursorPos(position);


				ImGui::PushClipRect(windowPosition + ImGui::GetCursorPos(),
									windowPosition + ImGui::GetCursorPos() +
										ImVec2(ImGui::GetContentRegionAvail().x, profilerHeight),
									true);

				applicationContext_->gpuGraph_.RenderTimings(graphWidth, legendWidth, graphHeight, frameOffset);
				ImGui::PopClipRect();
			}
		}
	}
	auto& io = ImGui::GetIO();
	animator_.animate(io.DeltaTime);
}

auto VolumeView::onResize() -> void
{
	deinitializeGraphicsResources();
	initializeGraphicsResources();
}

auto VolumeView::setRenderVolume(b3d::renderer::RendererBase* renderer,
								 b3d::renderer::RenderingDataWrapper* renderingData) -> void
{
	renderer_ = renderer;
	renderingData_ = renderingData;
}

auto VolumeView::setInternalRenderingResolutionScale(const float scale) -> void
{
	internalResolutionScale_ = scale;
	assert(glm::length(internalGraphicsResources_.internalVolumeTextureSize) > 0.001f);
	glfwMakeContextCurrent(applicationContext_->mainWindowHandle_);
	deinitializeInternalGraphicsResources();
	initializeInternalGraphicsResources();
}

auto VolumeView::drawGizmos(const CameraMatrices& cameraMatrices, const glm::vec2& position,
							const glm::vec2& size) const -> void
{
	constexpr auto currentGizmoMode = ImGuizmo::LOCAL;
	ImGuizmo::SetDrawlist(); // TODO: set before if statement, otherwise it can lead to crashes
	ImGuizmo::SetRect(position.x, position.y, size.x, size.y);
	if (currentGizmoOperation_ != GizmoOperationFlagBits::none)
	{

		auto gizmoOperation = ImGuizmo::OPERATION{};
		if (currentGizmoOperation_.containsBit(GizmoOperationFlagBits::rotate))
		{
			gizmoOperation = gizmoOperation | ImGuizmo::ROTATE;
		}
		if (currentGizmoOperation_.containsBit(GizmoOperationFlagBits::translate))
		{
			gizmoOperation = gizmoOperation | ImGuizmo::TRANSLATE;
		}
		if (currentGizmoOperation_.containsBit(GizmoOperationFlagBits::scale))
		{
			gizmoOperation = gizmoOperation | ImGuizmo::SCALE;
		}


		for (const auto transform : applicationContext_->getGizmoHelper()->getTransforms())
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
								 reinterpret_cast<const float*>(&cameraMatrices.projection), gizmoOperation,
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

	for (const auto& [bound, transform, worldTransform] : applicationContext_->getGizmoHelper()->getBoundTransforms())
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

		const auto bounds = std::array{ -halfSize.x, -halfSize.y, -halfSize.z, halfSize.x, halfSize.y, halfSize.z };

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

auto VolumeView::initializeGraphicsResources() -> void
{
	glfwMakeContextCurrent(applicationContext_->mainWindowHandle_);
	initializeViewGraphicsResources();
	initializeInternalGraphicsResources();
}

auto VolumeView::deinitializeGraphicsResources() -> void
{
	glfwMakeContextCurrent(applicationContext_->mainWindowHandle_);
	deinitializeInternalGraphicsResources();
	deinitializeViewGraphicsResources();
}

auto VolumeView::initializeViewGraphicsResources() -> void
{
	GL_CALL(glCreateFramebuffers(1, &viewGraphicsResources_.viewFbo.nativeHandle));
	GL_CALL(glCreateTextures(GL_TEXTURE_2D, 1, &viewGraphicsResources_.viewFbo.colorAttachment));
	GL_CALL(glTextureParameteri(viewGraphicsResources_.viewFbo.colorAttachment, GL_TEXTURE_BASE_LEVEL, 0));
	GL_CALL(glTextureParameteri(viewGraphicsResources_.viewFbo.colorAttachment, GL_TEXTURE_MAX_LEVEL, 0));
	GL_CALL(glTextureStorage2D(viewGraphicsResources_.viewFbo.colorAttachment, 1, GL_RGBA8,
							   static_cast<GLsizei>(viewportSize_.x), static_cast<GLsizei>(viewportSize_.y)));
	GL_CALL(glNamedFramebufferTexture(viewGraphicsResources_.viewFbo.nativeHandle, GL_COLOR_ATTACHMENT0,
									  viewGraphicsResources_.viewFbo.colorAttachment, 0));
}

auto VolumeView::deinitializeViewGraphicsResources() -> void
{
	GL_CALL(glDeleteTextures(1, &viewGraphicsResources_.viewFbo.colorAttachment));
	GL_CALL(glDeleteFramebuffers(1, &viewGraphicsResources_.viewFbo.nativeHandle));
}

auto VolumeView::initializeInternalGraphicsResources() -> void
{
	internalGraphicsResources_.internalVolumeTextureSize =
		glm::vec2{ viewportSize_.x, viewportSize_.y } * internalResolutionScale_;
	assert(glm::length(internalGraphicsResources_.internalVolumeTextureSize) > 0.001f);

	GL_CALL(glCreateTextures(GL_TEXTURE_2D, 1, &internalGraphicsResources_.internalVolumeTexture));
	GL_CALL(glTextureParameteri(internalGraphicsResources_.internalVolumeTexture, GL_TEXTURE_BASE_LEVEL, 0));
	GL_CALL(glTextureParameteri(internalGraphicsResources_.internalVolumeTexture, GL_TEXTURE_MAX_LEVEL, 0));
	GL_CALL(glTextureParameteri(internalGraphicsResources_.internalVolumeTexture, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL_CALL(glTextureParameteri(internalGraphicsResources_.internalVolumeTexture, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL_CALL(glTextureStorage2D(internalGraphicsResources_.internalVolumeTexture, 1, GL_RGBA8,
							   static_cast<GLsizei>(internalGraphicsResources_.internalVolumeTextureSize.x),
							   static_cast<GLsizei>(internalGraphicsResources_.internalVolumeTextureSize.y)));

	OWL_CUDA_CHECK(
		cudaMalloc(&internalGraphicsResources_.volumeTexturePointer,
				   static_cast<size_t>(internalGraphicsResources_.internalVolumeTextureSize.x *
									   internalGraphicsResources_.internalVolumeTextureSize.y * sizeof(uint32_t))));

	if (internalGraphicsResources_.cuInternalVolumeTexture)
	{
		OWL_CUDA_CHECK(cudaGraphicsUnregisterResource(internalGraphicsResources_.cuInternalVolumeTexture));
		internalGraphicsResources_.cuInternalVolumeTexture = 0;
	}

	OWL_CUDA_CHECK(cudaGraphicsGLRegisterImage(&internalGraphicsResources_.cuInternalVolumeTexture,
											   internalGraphicsResources_.internalVolumeTexture, GL_TEXTURE_2D, 0));
}

auto VolumeView::deinitializeInternalGraphicsResources() -> void
{
	GL_CALL(glDeleteTextures(1, &internalGraphicsResources_.internalVolumeTexture));

	if (internalGraphicsResources_.cuInternalVolumeTexture)
	{
		OWL_CUDA_CHECK(cudaGraphicsUnregisterResource(internalGraphicsResources_.cuInternalVolumeTexture));
		internalGraphicsResources_.cuInternalVolumeTexture = 0;
	}

	if (internalGraphicsResources_.volumeTexturePointer)
	{
		OWL_CUDA_CHECK(cudaFree(internalGraphicsResources_.volumeTexturePointer));
		internalGraphicsResources_.volumeTexturePointer = 0;
	}
}

auto VolumeView::renderVolume() -> void
{

	const auto width = static_cast<int>(viewportSize_.x);
	const auto height = static_cast<int>(viewportSize_.y);
	const auto [view, projection, viewProjection] = computeViewProjectionMatrixFromCamera(camera_, width, height);

	if (renderingData_ && renderer_ && internalGraphicsResources_.cuInternalVolumeTexture)
	{
		const auto cam = b3d::renderer::Camera{ .origin = owl_cast(camera_.getFrom()),
												.at = owl_cast(camera_.getAt()),
												.up = owl_cast(camera_.getUp()),
												.cosFoV = camera_.getCosFovY(),
												.FoV = glm::radians(camera_.getFovYInDegrees()) };

		renderingData_->data.view = b3d::renderer::View{
			.cameras = { cam, cam },
			.mode = b3d::renderer::RenderMode::mono,
		};

		renderingData_->data.renderTargets = b3d::renderer::RenderTargets{
			.colorRt = { internalGraphicsResources_.cuInternalVolumeTexture,
						 { static_cast<uint32_t>(internalGraphicsResources_.internalVolumeTextureSize.x),
						   static_cast<uint32_t>(internalGraphicsResources_.internalVolumeTextureSize.y), 1 } },
			.minMaxRt = { internalGraphicsResources_.cuInternalVolumeTexture,
						  { static_cast<uint32_t>(internalGraphicsResources_.internalVolumeTextureSize.x),
							static_cast<uint32_t>(internalGraphicsResources_.internalVolumeTextureSize.y), 1 } },
		};

		renderer_->render();
	}

	GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, viewGraphicsResources_.viewFbo.nativeHandle));
	const auto colorClearValue = std::array{ 0.0f, 0.0f, 0.0f, 1.0f };
	GL_CALL(
		glClearNamedFramebufferfv(viewGraphicsResources_.viewFbo.nativeHandle, GL_COLOR, 0, colorClearValue.data()));

	const auto& r1 = applicationContext_->getGlGpuTimers().record("Fullscreen Quad Pass");
	r1.start();
	fullscreenTexturePass_->setViewport(width, height);
	fullscreenTexturePass_->setSourceTexture(internalGraphicsResources_.internalVolumeTexture);
	fullscreenTexturePass_->execute();
	r1.stop();

	if (viewerSettings_.enableGridFloor)
	{
		const auto& record = applicationContext_->getGlGpuTimers().record("Grid Floor Pass");
		record.start();
		infiniteGridPass_->setViewProjectionMatrix(viewProjection);
		infiniteGridPass_->setViewport(width, height);
		infiniteGridPass_->setGridColor(
			glm::vec3{ viewerSettings_.gridColor[0], viewerSettings_.gridColor[1], viewerSettings_.gridColor[2] });
		infiniteGridPass_->execute();
		record.stop();
	}

	if (viewerSettings_.enableDebugDraw)
	{
		const auto& record = applicationContext_->getGlGpuTimers().record("Debug Draw Pass");
		record.start();
		debugDrawPass_->setViewProjectionMatrix(viewProjection);
		debugDrawPass_->setViewport(width, height);
		debugDrawPass_->setLineWidth(viewerSettings_.lineWidth);
		debugDrawPass_->execute();
		record.stop();
	}

	GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

auto VolumeView::demoMode(const bool enable) -> void
{
	static auto backup = viewerSettings_;
	demoModeEnabled_ = enable;

	if (enable)
	{
		viewerSettings_.enableDebugDraw = false;
		viewerSettings_.enableGridFloor = false;
		viewerSettings_.enableControlToolBar = false;
		animator_.start();
	}
	else
	{
		viewerSettings_ = backup;
		animator_.pause();
	}
}
