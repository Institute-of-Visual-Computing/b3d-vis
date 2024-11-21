#pragma once

#include "Camera.h"
class CameraController
{
public:
	virtual auto update() -> void = 0;
	auto setCamera(Camera* camera)
	{
		camera_ = camera;
	}

	virtual ~CameraController()
	{
	}

protected:
	Camera* camera_{nullptr};
};

class FirstPersonCameraController final : public CameraController
{
private:
	bool moveCameraFaster_{ false };

public:
	auto update() -> void override;

private:
};


class OrbitCameraController final : public CameraController
{
public:
	auto update() -> void override;
};
//
//class AnimatedCameraController : public CameraController
//{
//public:
//	auto animatePosition(const glm::vec3& position) -> void
//	{
//		// camera_->position_ = position;
//	}
//	auto animateOrientation(const glm::quat& orientation) -> void
//	{
//	}
//	auto enableTwinning() -> void
//	{
//		isTwinningEnabled_ = true;
//	}
//	auto disableTwinning() -> void
//	{
//		isTwinningEnabled_ = false;
//	}
//
//private:
//	bool isTwinningEnabled_{ false };
//	glm::vec3 desiredPosition_{};
//	//glm::quat desiredOrientation_{};
//};
