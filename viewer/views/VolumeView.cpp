#include "VolumeView.h"
#include "../GizmoHelper.h"

#include "../GLUtils.h"

#include <GLFW/glfw3.h>

#include <ImGuizmo.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>

#include <IconsFontAwesome6Brands.h>
#include <IconsLucide.h>

#include "../ApplicationContext.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <owl/helper/cuda.h>

#include <RendererBase.h>

#include "../passes/DebugDrawPass.h"
#include "../passes/FullscreenTexturePass.h"
#include "../passes/InfinitGridPass.h"
#include "../InteropUtils.h"

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
							 WindowFlagBits::noTitleBar | WindowFlagBits::noUndocking | WindowFlagBits::hideTabBar |
								 WindowFlagBits::noCollapse)
{
	gizmoHelper_ = std::make_shared<GizmoHelper>();
	//initializeGraphicsResources();

	debugDrawList_ = std::make_unique<DebugDrawList>();
	fullscreenTexturePass_ = std::make_unique<FullscreenTexturePass>();
	InfinitGridPass_ = std::make_unique<InfinitGridPass>();
	debugDrawPass_ = std::make_unique<DebugDrawPass>(debugDrawList_.get());

	camera_.setOrientation(glm::vec3(1.0, 1.0, 1.0), glm::vec3(0.0, 0.0, 0.0), camera_.getUp(),
						  camera_.getFovYInDegrees());
}

VolumeView::~VolumeView()
{
	deinitializeGraphicsResources();
}

auto VolumeView::onDraw() -> void
{
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


	ImVec2 p = ImGui::GetCursorScreenPos();
	ImGui::SetNextItemAllowOverlap();
	ImGui::InvisibleButton("##volumeViewport", viewportSize_,
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
	ImGui::Image((ImTextureID)graphicsResources_.framebufferTexture, viewportSize_, { 0.0f, 1.0f }, { 1.0f, 0.0f });

	if (viewerSettings_.enableDebugDraw)
	{
		ImGui::SetNextItemAllowOverlap();
		ImGui::SetCursorScreenPos(p);
		const auto cameraMatrices = computeViewProjectionMatrixFromCamera(camera_, viewportSize_.x, viewportSize_.y);
		drawGizmos(cameraMatrices, glm::vec2{ p.x, p.y }, glm::vec2{ viewportSize_.x, viewportSize_.y });
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
	}
}

auto VolumeView::onResize() -> void
{
	deinitializeGraphicsResources();
	initializeGraphicsResources();
}

auto VolumeView::renderVolume(b3d::renderer::RendererBase* renderer,
							  b3d::renderer::RenderingDataWrapper* renderingData) -> void
{
	const auto width = viewportSize_.x;
	const auto height = viewportSize_.y;
	const auto cameraMatrices = computeViewProjectionMatrixFromCamera(camera_, width, height);

	GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, graphicsResources_.framebuffer));
	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);



	const auto cam = b3d::renderer::Camera{ .origin = owl_cast(camera_.getFrom()),
											.at = owl_cast(camera_.getAt()),
											.up = owl_cast(camera_.getUp()),
											.cosFoV = camera_.getCosFovY(),
											.FoV = glm::radians(camera_.getFovYInDegrees()) };

	renderingData->data.view = b3d::renderer::View{
		.cameras = { cam, cam },
		.mode = b3d::renderer::RenderMode::mono,
	};

	renderingData->data.renderTargets = b3d::renderer::RenderTargets{
		.colorRt = { graphicsResources_.cuFramebufferTexture,
					 { static_cast<uint32_t>(graphicsResources_.framebufferSize.x),
					   static_cast<uint32_t>(graphicsResources_.framebufferSize.y), 1 } },
		.minMaxRt = { graphicsResources_.cuFramebufferTexture,
					  { static_cast<uint32_t>(graphicsResources_.framebufferSize.x),
						static_cast<uint32_t>(graphicsResources_.framebufferSize.y), 1 } },
	};

	renderer->render();



	fullscreenTexturePass_->setViewport(width, height);
	fullscreenTexturePass_->setSourceTexture(graphicsResources_.framebufferTexture);
	fullscreenTexturePass_->execute();


	if (viewerSettings_.enableGridFloor)
	{
		InfinitGridPass_->setViewProjectionMatrix(cameraMatrices.viewProjection);
		InfinitGridPass_->setViewport(width, height);
		InfinitGridPass_->setGridColor(
			glm::vec3{ viewerSettings_.gridColor[0], viewerSettings_.gridColor[1], viewerSettings_.gridColor[2] });
		InfinitGridPass_->execute();
	}

	if (viewerSettings_.enableDebugDraw)
	{
		debugDrawPass_->setViewProjectionMatrix(cameraMatrices.viewProjection);
		debugDrawPass_->setViewport(width, height);
		debugDrawPass_->setLineWidth(viewerSettings_.lineWidth);
		debugDrawPass_->execute();
	}

	GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

auto VolumeView::drawGizmos(const CameraMatrices& cameraMatrices, const glm::vec2& position, const glm::vec2& size)
	-> void
{
	const auto currentGizmoMode = ImGuizmo::LOCAL;
	ImGuizmo::SetDrawlist(); // TODO: set before if statement, otherwise it can lead to crashes
	ImGuizmo::SetRect(position.x, position.y, size.x, size.y);
	if (currentGizmoOperation_ != GizmoOperationFlagBits::none)
	{

		auto guizmoOperation = ImGuizmo::OPERATION{};
		if (currentGizmoOperation_.containsBit(GizmoOperationFlagBits::rotate))
		{
			guizmoOperation = guizmoOperation | ImGuizmo::ROTATE;
		}
		if (currentGizmoOperation_.containsBit(GizmoOperationFlagBits::translate))
		{
			guizmoOperation = guizmoOperation | ImGuizmo::TRANSLATE;
		}
		if (currentGizmoOperation_.containsBit(GizmoOperationFlagBits::scale))
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

auto VolumeView::initializeGraphicsResources() -> void
{
	glfwMakeContextCurrent(applicationContext_->mainWindowHandle_);


	if (graphicsResources_.framebufferPointer)
	{
		OWL_CUDA_CHECK(cudaFree(graphicsResources_.framebufferPointer));
	}
	OWL_CUDA_CHECK(cudaMallocManaged(&graphicsResources_.framebufferPointer, viewportSize_.x * viewportSize_.y * sizeof(uint32_t)));

	graphicsResources_.framebufferSize = glm::vec2{ viewportSize_.x, viewportSize_.y };
	if (graphicsResources_.framebufferTexture == 0)
	{
		GL_CALL(glGenTextures(1, &graphicsResources_.framebufferTexture));
	}
	else
	{
		if (graphicsResources_.cuFramebufferTexture)
		{
			OWL_CUDA_CHECK(cudaGraphicsUnregisterResource(graphicsResources_.cuFramebufferTexture));
			graphicsResources_.cuFramebufferTexture = 0;
		}
	}

	GL_CALL(glBindTexture(GL_TEXTURE_2D, graphicsResources_.framebufferTexture));
	GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, viewportSize_.x, viewportSize_.y, 0, GL_RGBA, GL_UNSIGNED_BYTE,
						 nullptr));
	GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

	OWL_CUDA_CHECK(cudaGraphicsGLRegisterImage(&graphicsResources_.cuFramebufferTexture,
											   graphicsResources_.framebufferTexture, GL_TEXTURE_2D, 0));

	GL_CALL(glGenFramebuffers(1, &graphicsResources_.framebuffer));

	GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, graphicsResources_.framebuffer));
	GL_CALL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
								   graphicsResources_.framebufferTexture, 0));
	GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

auto VolumeView::deinitializeGraphicsResources() -> void
{
}


