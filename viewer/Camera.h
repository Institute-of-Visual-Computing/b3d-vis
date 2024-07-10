#pragma once
#include <glm/glm.hpp>

class Camera final
{
public:
	[[nodiscard]] inline float getFovYInDegrees() const
	{
		return fovYInDegrees_;
	}
	[[nodiscard]] inline float getCosFovY() const
	{
		return glm::cos(glm::radians(fovYInDegrees_));
	}

	[[nodiscard]] inline auto getFrom() const -> glm::vec3
	{
		return position_;
	}

	[[nodiscard]] inline auto getAt() const -> glm::vec3
	{
		return position_ + forward_;
	}
	[[nodiscard]] inline auto getUp() const -> glm::vec3
	{
		return glm::vec3{0.0f,1.0f,0.0f};// up_;
	};

	inline auto setOrientation(const glm::vec3& origin, const glm::vec3& interest, const glm::vec3& up,
							   float fovYInDegrees) -> void
	{
		fovYInDegrees_ = fovYInDegrees;
		position_ = origin;
		up_ = up;
		forward_ = (interest == origin) ? glm::vec3(0, 0, 1) : normalize(interest - origin);
		right_ = glm::normalize(glm::cross(up, forward_));

		up_ = glm::normalize(glm::cross(forward_, right_));
		// forceUpFrame();
	}

public:
	glm::vec3 up_{ 0, 1, 0 };
	glm::vec3 forward_;
	glm::vec3 right_;
	glm::vec3 position_;
	float fovYInDegrees_{ 60.f };

	float movementSpeedScale_{ 1.0f };
};
