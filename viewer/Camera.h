#pragma once
#include <glm/glm.hpp>

class Camera final
{
public:
	[[nodiscard]] auto getFovYInDegrees() const -> float
	{
		return fovYInDegrees_;
	}
	[[nodiscard]] auto getCosFovY() const -> float
	{
		return glm::cos(glm::radians(fovYInDegrees_));
	}

	[[nodiscard]] auto getFrom() const -> glm::vec3
	{
		return position_;
	}

	[[nodiscard]] inline auto getAt() const -> glm::vec3
	{
		return position_ + forward_;
	}
	[[nodiscard]] static auto getUp() -> glm::vec3
	{
		return glm::vec3{0.0f,1.0f,0.0f};;
	};

	inline auto setOrientation(const glm::vec3& origin, const glm::vec3& interest, const glm::vec3& up,
							   const float fovYInDegrees) -> void
	{
		fovYInDegrees_ = fovYInDegrees;
		position_ = origin;
		up_ = up;
		forward_ = (interest == origin) ? glm::vec3(0, 0, 1) : normalize(interest - origin);
		right_ = glm::normalize(glm::cross(up, forward_));

		up_ = glm::normalize(glm::cross(forward_, right_));
	}

public:
	glm::vec3 up_{ 0, 1, 0 };
	glm::vec3 forward_;
	glm::vec3 right_;
	glm::vec3 position_;
	float fovYInDegrees_{ 60.f };

	float movementSpeedScale_{ 1.0f };
};
