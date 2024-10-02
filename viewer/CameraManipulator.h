#pragma once

class CameraManipulator final
{
private:
	const float pixelPerMove_{10.0f};
	const float degreesPerDragFraction_{150.0f};
	const float kbdRotateDegrees_{10.0f};
};
