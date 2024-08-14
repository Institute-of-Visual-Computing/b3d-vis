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

#include "GLUtils.h"
#include "GizmoHelper.h"
#include "InteropUtils.h"
#include "framework/ApplicationContext.h"
#include "passes/DebugDrawPass.h"
#include "passes/FullscreenTexturePass.h"
#include "passes/InfinitGridPass.h"

namespace animation
{

	auto fastAtan(const float x) -> float
	{
		const auto z = fabs(x);
		const auto w = z > 1.0f ? 1.0f / z : z;
		const float y = (M_PI / 4.0f) * w - w * (w - 1) * (0.2447f + 0.0663f * w);
		return copysign(z > 1.0f ? M_PI / 2.0 - y : y, x);
	}

	auto fastNegExp(const float x) -> float
	{
		return 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
	}

	auto squaref(const float x) -> float
	{
		return x * x;
	}

	auto springDamperExact(float& x, float& v, const float xGoal, const float vGoal, const float stiffness,
						   const float damping, const float dt, const float eps = 1e-5f) -> void
	{
		const auto g = xGoal;
		const auto q = vGoal;
		const auto s = stiffness;
		const auto d = damping;
		const auto c = g + (d * q) / (s + eps);
		const auto y = d / 2.0f;
		const auto w = sqrtf(s - (d * d) / 4.0f);
		auto j = sqrtf(squaref(v + y * (x - c)) / (w * w + eps) + squaref(x - c));
		const auto p = fastAtan((v + (x - c) * y) / (-(x - c) * w + eps));

		j = (x - c) > 0.0f ? j : -j;

		const auto eydt = fastNegExp(y * dt);

		x = j * eydt * cosf(w * dt + p) + c;
		v = -y * j * eydt * cosf(w * dt + p) - w * j * eydt * sinf(w * dt + p);
	}


	using PropertyAnimation = std::function<void(float, float)>;

	class PropertyAnimator final
	{
	public:
		auto addPropertyAnimation(const PropertyAnimation& animation)
		{
			animations_.push_back(animation);
		}

		auto animate(const float delta) -> void
		{
			if (isRunning_)
			{
				globalTime_ += delta;
				for (auto& animation : animations_)
				{
					animation(globalTime_, delta);
				}
			}
		}

		auto reset() -> void
		{
			animations_.clear();
		}
		auto start() -> void
		{
			isRunning_ = true;
		}
		auto stop() -> void
		{
			pause();
			globalTime_ = 0.0f;
		}
		auto pause() -> void
		{
			isRunning_ = false;
		}

		auto isRunning() const -> bool
		{
			return isRunning_;
		}

	private:
		std::vector<PropertyAnimation> animations_{};
		float globalTime_{ 0.0f };

		bool isRunning_{ false };
	};
} // namespace animation

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

	animation::PropertyAnimator animator;
	bool demoModeEnabled = false;
} // namespace

VolumeView::VolumeView(ApplicationContext& appContext, Dockspace* dockspace)
	: DockableWindowViewBase(appContext, "Volume Viewport", dockspace,
							 WindowFlagBits::noTitleBar | WindowFlagBits::noUndocking | WindowFlagBits::hideTabBar |
								 WindowFlagBits::noCollapse)
{
	fullscreenTexturePass_ = std::make_unique<FullscreenTexturePass>();
	infinitGridPass_ = std::make_unique<InfinitGridPass>();
	debugDrawPass_ = std::make_unique<DebugDrawPass>(applicationContext_->getDrawList().get());

	camera_.setOrientation(glm::vec3(1.0, 1.0, 1.0), glm::vec3(0.0, 0.0, 0.0), camera_.getUp(),
						   camera_.getFovYInDegrees());


	struct CameraFlyAroundAnimationSetting
	{
		glm::vec3 origin{};
		glm::vec3 target{ 0.0f, 0.0f, 0.0f };
		float height{ 1.0f };
		float radius{ 1.0f };
		float stiffness{ 20.0f };
		float dumping{ 6.0f };
	};

	auto flyAnimationSettings = CameraFlyAroundAnimationSetting{};
	flyAnimationSettings.radius = 3.0f;

	animator.addPropertyAnimation(
		[&, flyAnimationSettings](const float t, const float dt)
		{
			auto cameraAnimationPathPosition = [=](const float x) -> glm::vec3
			{
				const auto radius = flyAnimationSettings.radius;
				const auto height = flyAnimationSettings.height;
				const auto& origin = flyAnimationSettings.origin;
				const auto sin = glm::sin(x);
				const auto cos = glm::cos(x);

				const auto offset = glm::vec3{ sin * radius, height, cos * radius };

				return origin + offset;
			};

			const auto lastPosition = cameraAnimationPathPosition(t - dt);
			const auto position = cameraAnimationPathPosition(t);

			const auto targetVelocity = position - lastPosition;

			const auto lastTargetLookDirection = glm::normalize(flyAnimationSettings.origin - lastPosition);
			const auto targetLookDirection = glm::normalize(flyAnimationSettings.origin - position);
			const auto targetLookDirectionVelocity = targetLookDirection - lastTargetLookDirection;

			const auto cameraVelocity = position - camera_.position_;
			const auto lastCameraPosition = camera_.position_;


			{

				auto x = camera_.position_.x;
				auto y = camera_.position_.y;
				auto z = camera_.position_.z;

				auto vx = cameraVelocity.x;
				auto vy = cameraVelocity.y;
				auto vz = cameraVelocity.z;


				animation::springDamperExact(x, vx, position.x, targetVelocity.x, flyAnimationSettings.stiffness,
											 flyAnimationSettings.dumping, dt);
				animation::springDamperExact(y, vy, position.y, targetVelocity.y, flyAnimationSettings.stiffness,
											 flyAnimationSettings.dumping, dt);
				animation::springDamperExact(z, vz, position.z, targetVelocity.z, flyAnimationSettings.stiffness,
											 flyAnimationSettings.dumping, dt);
				camera_.position_ = glm::vec3{ x, y, z };
			}

			{

				const auto lastForwardTarget =
					glm::dot(glm::normalize(flyAnimationSettings.target - lastPosition), camera_.forward_);
				const auto forwardTarget =
					glm::dot(glm::normalize(flyAnimationSettings.target - camera_.position_), camera_.forward_);

				auto v = forwardTarget - lastForwardTarget;
				auto p = forwardTarget;
				animation::springDamperExact(p, v, 0.0f, 0.0f, flyAnimationSettings.stiffness,
											 flyAnimationSettings.dumping, dt);


				const auto rotation = glm::rotate(
					glm::identity<glm::mat4>(), p,
					glm::normalize(
						glm::cross(glm::normalize(flyAnimationSettings.target - camera_.position_), camera_.forward_)));

				const auto target = glm::normalize(glm::vec3(rotation * glm::vec4(camera_.forward_, 0.0f)));


				const auto interest = camera_.position_ - target;
				camera_.setOrientation(camera_.position_, flyAnimationSettings.target, camera_.up_, 60);
			}
		});

	applicationContext_->addMenuToggleAction(
		demoModeEnabled, [&](const bool isOn) { demoMode(isOn); }, "Help", "Demo Mode", "F2", std::nullopt, 0);
}

VolumeView::~VolumeView()
{
	deinitializeGraphicsResources();
}


auto VolumeView::onDraw() -> void
{
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


	ImVec2 p = ImGui::GetCursorScreenPos();
	ImGui::SetNextItemAllowOverlap();
	ImGui::InvisibleButton("##volumeViewport", viewportSize_,
						   ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);

	const auto viewportIsFocused = ImGui::IsItemFocused();
	static auto moveCameraFaster = false;
	auto& io = ImGui::GetIO();


	class CameraController
	{
	public:
		virtual ~CameraController()
		{
		}

	private:
		Camera* camera_{};
	};

	class FirstPersonCameraController : public CameraController
	{
	private:
	};

	class OrbitCameraController : public CameraController
	{
	};

	class AnimatedCameraController : public CameraController
	{
	public:
		auto animatePosition(const glm::vec3& position) -> void
		{
			// camera_->position_ = position;
		}
		auto animateOrientation(const glm::quat& orientation) -> void
		{
		}
		auto enableTwinning() -> void
		{
			isTwinningEnabled_ = true;
		}
		auto disableTwinning() -> void
		{
			isTwinningEnabled_ = false;
		}

	private:
		bool isTwinningEnabled_{ false };
		glm::vec3 desiredPosition_{};
		glm::quat desiredOrientation_{};
	};


	constexpr auto fastSpeed = 25.0f;
	constexpr auto cameraMoveVelocity = 0.0f;
	auto cameraMoveAcceleration = glm::vec3{ 0 };
	constexpr auto maxCameraMoveAcceleration = 1.0f;
	static auto accelerationExpire = 0.0;
	if (viewportIsFocused)
	{

		if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
		{
			moveCameraFaster = true;
		}

		if (ImGui::IsKeyReleased(ImGuiKey_LeftShift))
		{
			moveCameraFaster = false;
		}

		if (ImGui::IsKeyDown(ImGuiKey_W))
		{
			cameraMoveAcceleration =
				camera_.forward_ * camera_.movementSpeedScale_ * (moveCameraFaster ? fastSpeed : 1.0f);
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
					constexpr auto sensitivity = 0.1f;
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
	}

	if (ImGui::IsKeyPressed(ImGuiKey_F2, false))
	{
		demoMode(!animator.isRunning());
	}
	animator.animate(io.DeltaTime);
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
	OWL_CUDA_CHECK(cudaMallocManaged(&graphicsResources_.framebufferPointer,
									 viewportSize_.x * viewportSize_.y * sizeof(uint32_t)));

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

auto VolumeView::renderVolume() -> void
{

	const auto width = viewportSize_.x;
	const auto height = viewportSize_.y;
	const auto cameraMatrices = computeViewProjectionMatrixFromCamera(camera_, width, height);

	GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, graphicsResources_.framebuffer));
	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);


	if (renderingData_ && renderer_ && graphicsResources_.cuFramebufferTexture)
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
			.colorRt = { graphicsResources_.cuFramebufferTexture,
						 { static_cast<uint32_t>(graphicsResources_.framebufferSize.x),
						   static_cast<uint32_t>(graphicsResources_.framebufferSize.y), 1 } },
			.minMaxRt = { graphicsResources_.cuFramebufferTexture,
						  { static_cast<uint32_t>(graphicsResources_.framebufferSize.x),
							static_cast<uint32_t>(graphicsResources_.framebufferSize.y), 1 } },
		};

		renderer_->render();
	}

	fullscreenTexturePass_->setViewport(width, height);
	fullscreenTexturePass_->setSourceTexture(graphicsResources_.framebufferTexture);
	fullscreenTexturePass_->execute();


	if (viewerSettings_.enableGridFloor)
	{
		infinitGridPass_->setViewProjectionMatrix(cameraMatrices.viewProjection);
		infinitGridPass_->setViewport(width, height);
		infinitGridPass_->setGridColor(
			glm::vec3{ viewerSettings_.gridColor[0], viewerSettings_.gridColor[1], viewerSettings_.gridColor[2] });
		infinitGridPass_->execute();
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

auto VolumeView::demoMode(const bool enable) -> void
{
	static auto backup = viewerSettings_;
	demoModeEnabled = enable;

	if (enable)
	{
		viewerSettings_.enableDebugDraw = false;
		viewerSettings_.enableGridFloor = false;
		viewerSettings_.enableControlToolBar = false;
		animator.start();
	}
	else
	{
		viewerSettings_ = backup;
		animator.pause();
	}
}
