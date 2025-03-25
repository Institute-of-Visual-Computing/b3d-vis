#include "CameraControllers.h"

//#include "ImGuizmo.h"
#include "glm/ext/matrix_transform.hpp"
#include "imgui.h"


auto FirstPersonCameraController::update() -> void
{
	const auto viewportIsFocused = ImGui::IsItemFocused();
	auto& io = ImGui::GetIO();


	constexpr auto fastSpeed = 25.0f;
	auto cameraMoveAcceleration = glm::vec3{ 0 };
	if (viewportIsFocused)
	{

		if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
		{
			moveCameraFaster_ = true;
		}

		if (ImGui::IsKeyReleased(ImGuiKey_LeftShift))
		{
			moveCameraFaster_ = false;
		}

		if (ImGui::IsKeyDown(ImGuiKey_W))
		{
			cameraMoveAcceleration =
				camera_->forward_ * camera_->movementSpeedScale_ * (moveCameraFaster_ ? fastSpeed : 1.0f);
		}
		if (ImGui::IsKeyDown(ImGuiKey_S))
		{
			cameraMoveAcceleration =
				-camera_->forward_ * camera_->movementSpeedScale_ * (moveCameraFaster_ ? fastSpeed : 1.0f);
		}
		if (ImGui::IsKeyDown(ImGuiKey_A))
		{
			cameraMoveAcceleration = -glm::normalize(glm::cross(camera_->forward_, camera_->getUp())) *
				camera_->movementSpeedScale_ * (moveCameraFaster_ ? fastSpeed : 1.0f);
		}
		if (ImGui::IsKeyDown(ImGuiKey_D))
		{
			cameraMoveAcceleration = glm::normalize(glm::cross(camera_->forward_, camera_->getUp())) *
				camera_->movementSpeedScale_ * (moveCameraFaster_ ? fastSpeed : 1.0f);
		}

		// orbital control
	}

	auto delta = io.MouseDelta;
	delta.x *= -1.0;

	if (ImGui::IsItemHovered())
	{
		const auto wheelValue = io.MouseWheel;
		if (wheelValue != 0.0f)
		{
			cameraMoveAcceleration = camera_->forward_ * camera_->movementSpeedScale_ * wheelValue * fastSpeed;
		}
	}

	if (ImGui::IsItemActive())
	{
		//if (!ImGuizmo::IsUsing())
		{
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
			{
				const auto right = glm::normalize(glm::cross(camera_->forward_, camera_->getUp()));
				cameraMoveAcceleration += -glm::normalize(glm::cross(camera_->forward_, right)) *
					camera_->movementSpeedScale_ * (moveCameraFaster_ ? fastSpeed : 1.0f) * delta.y;

				cameraMoveAcceleration +=
					right * camera_->movementSpeedScale_ * (moveCameraFaster_ ? fastSpeed : 1.0f) * delta.x;
			}
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
			{
				const auto up = camera_->getUp();
				const auto right = glm::normalize(glm::cross(up, camera_->forward_));

				const auto f = glm::normalize(camera_->forward_ + right * delta.y + up * delta.x);

				auto rotationAxis = glm::normalize(glm::cross(f, camera_->forward_));

				if (glm::length(rotationAxis) >= 0.001f)
				{
					constexpr auto sensitivity = 0.1f;
					const auto rotation =
						glm::rotate(glm::identity<glm::mat4>(),
									glm::radians(glm::length(glm::vec2{ delta.x, delta.y }) * sensitivity), f);
					camera_->forward_ = glm::normalize(glm::vec3(rotation * glm::vec4(camera_->forward_, 0.0f)));
					camera_->right_ = glm::normalize(glm::cross(camera_->forward_, up));
				}
			}


			if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
			{
				const auto up = camera_->getUp();
				const auto right = glm::normalize(glm::cross(up, camera_->forward_));

				const auto f = glm::normalize(camera_->forward_ + right * delta.y + up * delta.x);

				auto rotationAxis = glm::normalize(glm::cross(f, camera_->forward_));

				if (glm::length(rotationAxis) >= 0.001f)
				{
					constexpr auto sensitivity = 0.1f;
					const auto rotation =
						glm::rotate(glm::identity<glm::mat4>(),
									glm::radians(glm::length(glm::vec2{ delta.x, delta.y }) * sensitivity), f);
					camera_->forward_ = glm::normalize(glm::vec3(rotation * glm::vec4(camera_->forward_, 0.0f)));
					camera_->right_ = glm::normalize(glm::cross(camera_->forward_, up));
				}
			}
		}
	}

	camera_->position_ += cameraMoveAcceleration * io.DeltaTime;
}


auto OrbitCameraController::update() -> void
{
	auto& io = ImGui::GetIO();
	auto cameraMoveAcceleration = glm::vec3{ 0 };

	auto delta = io.MouseDelta;
	delta.x *= -1.0;

	if (ImGui::IsItemHovered())
	{
		const auto wheelValue = io.MouseWheel;
		if (wheelValue != 0.0f)
		{
			constexpr auto fastSpeed = 25.0f;
			cameraMoveAcceleration = camera_->forward_ * camera_->movementSpeedScale_ * wheelValue * fastSpeed;
		}
	}

	if (ImGui::IsItemActive())
	{
		//if (!ImGuizmo::IsUsing())
		{
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
			{
				const auto up = camera_->getUp();
				const auto right = glm::normalize(glm::cross(up, camera_->forward_));

				const auto rotationDirection = glm::normalize(right * delta.y + up * delta.x);
				const auto rotationAxis = glm::cross(rotationDirection, camera_->forward_);

				if (glm::length(rotationAxis) >= 0.001f)
				{
					constexpr auto focusPosition = glm::vec3{ 0.0f, 0.0f, 0.0f };

					const auto focusDistance = glm::distance(focusPosition, camera_->position_);
					constexpr auto sensitivity = 0.1f;
					const auto rotation = glm::rotate(
						glm::identity<glm::mat4>(),
						glm::radians(glm::length(glm::vec2{ delta.x, delta.y }) * sensitivity), rotationDirection);
					const auto newForward = glm::vec3(rotation * glm::vec4(-camera_->forward_, 0.0f));

					camera_->position_ =
						camera_->position_ + focusDistance * camera_->forward_ + focusDistance * newForward;

					camera_->forward_ = -newForward;
					camera_->right_ = glm::normalize(glm::cross(camera_->forward_, up));
				}
			}
		}
	}

	camera_->position_ += cameraMoveAcceleration * io.DeltaTime;
}
